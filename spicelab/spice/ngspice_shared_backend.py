"""ctypes-based libngspice backend implementing the shared adapter protocol."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from ctypes import util
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from ..engines.exceptions import EngineSharedLibraryNotFound
from ..io.readers import read_ngspice_raw
from .ngspice_shared import (
    ExternalSourceRequest,
    SharedAdapterResult,
    SharedCallbacks,
    SharedTranPoint,
)

__all__ = ["CtypesNgSpiceSharedBackend", "load_default_backend"]


SendCharCB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
SendStatCB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
ControlledExitCB = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p
)
SendDataCB = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
)
SendInitDataCB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
BGThreadCB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_bool, ctypes.c_void_p)


class VecValues(ctypes.Structure):
    """Mirror of ``vecvalues`` from ``sharedspice.h``."""

    _fields_ = [
        ("name", ctypes.c_char_p),
        ("creal", ctypes.c_double),
        ("cimag", ctypes.c_double),
        ("is_scale", ctypes.c_bool),
        ("is_complex", ctypes.c_bool),
    ]

    def real_value(self) -> float:
        return float(self.creal)


class VecValuesAll(ctypes.Structure):
    _fields_ = [
        ("veccount", ctypes.c_int),
        ("vecindex", ctypes.c_int),
        ("vecsa", ctypes.POINTER(ctypes.POINTER(VecValues))),
    ]


def _candidate_paths() -> Iterable[str]:
    env = os.environ.get("SPICELAB_NGSPICE_SHARED")
    if env:
        yield env
    home = Path.home()
    defaults = [
        "/opt/homebrew/lib/libngspice.dylib",
        "/usr/local/lib/libngspice.dylib",
        "/usr/lib/libngspice.dylib",
        "/usr/local/lib/libngspice.so",
        str(home / "lib" / "libngspice.dylib"),
        str(home / "lib" / "libngspice.so"),
    ]
    yield from defaults
    found = util.find_library("ngspice")
    if found:
        yield found
    # Fallback to default sonames
    yield from ["libngspice.dylib", "libngspice.so", "ngspice.dll"]


class CtypesNgSpiceSharedBackend:
    """Minimal ctypes binding around libngspice for transient analyses."""

    def __init__(self, library_path: str) -> None:
        self._lib = ctypes.CDLL(library_path)
        self._logs: list[str] = []
        self._active_callbacks: SharedCallbacks | None = None
        self._external_sources: Mapping[str, Callable[[ExternalSourceRequest], float]] = {}
        self._last_source_values: dict[str, float] = {}
        self._setup_signatures()
        self._install_callbacks()

    # Public API ---------------------------------------------------------

    def run(
        self,
        netlist: str,
        directives: Sequence[str],
        callbacks: SharedCallbacks | None,
        external_sources: Mapping[str, Callable[[ExternalSourceRequest], float]] | None,
    ) -> SharedAdapterResult:
        self._logs.clear()
        self._active_callbacks = callbacks
        self._external_sources = external_sources or {}
        summary = [f"{name}:{type(func).__name__}" for name, func in self._external_sources.items()]
        self._logs.append("external sources: " + ",".join(summary))
        self._last_source_values = {}
        if callbacks and callbacks.on_init:
            callbacks.on_init()

        with TemporaryDirectory(prefix="spicelab-ngshared-") as tmp_dir:
            tmp = Path(tmp_dir)
            raw_path = tmp / "out.raw"

            # Reset ngspice state and load circuit
            self._command("destroy all")
            self._command("set noaskquit")
            self._command(f"set rawfile={raw_path}")

            self._load_circuit(netlist)

            for directive in directives:
                cmd = directive.strip()
                if cmd.startswith("."):
                    cmd = cmd[1:]
                self._command(cmd)

            self._command("run")
            self._command("write")

            dataset = read_ngspice_raw(str(raw_path))

        self._command("reset")
        logs_copy = list(self._logs)

        if callbacks and callbacks.on_tran_point and "time" in dataset.coords:
            time_values = dataset.coords["time"].values
            data_names = list(dataset.data_vars)
            for idx, t in enumerate(time_values):
                point = {
                    name: float(dataset[name].values[idx])
                    for name in data_names
                    if dataset[name].values.ndim == 1
                }
                callbacks.on_tran_point(SharedTranPoint(time=float(t), values=point))

        if callbacks and callbacks.on_exit:
            callbacks.on_exit(0)

        self._active_callbacks = None
        self._external_sources = {}
        self._last_source_values = {}

        return SharedAdapterResult(
            dataset=dataset,
            log_lines=logs_copy,
            stdout="\n".join(logs_copy),
            stderr="",
        )

    # Internal helpers ---------------------------------------------------

    def _setup_signatures(self) -> None:
        self._lib.ngSpice_Init.restype = ctypes.c_int
        self._lib.ngSpice_Init.argtypes = [
            SendCharCB,
            SendStatCB,
            ControlledExitCB,
            SendDataCB,
            SendInitDataCB,
            BGThreadCB,
            ctypes.c_void_p,
        ]
        self._lib.ngSpice_Command.restype = ctypes.c_int
        self._lib.ngSpice_Command.argtypes = [ctypes.c_char_p]
        self._lib.ngSpice_Circ.restype = ctypes.c_int
        self._lib.ngSpice_Circ.argtypes = [ctypes.POINTER(ctypes.c_char_p)]

    def _install_callbacks(self) -> None:
        self._cb_send_char = SendCharCB(self._send_char)
        self._cb_send_stat = SendStatCB(self._send_char)
        self._cb_exit = ControlledExitCB(self._controlled_exit)
        self._cb_send_data = SendDataCB(self._send_data)
        self._cb_send_init = SendInitDataCB(self._send_init_data)
        self._cb_bg = BGThreadCB(self._bg_thread)
        rc = self._lib.ngSpice_Init(
            self._cb_send_char,
            self._cb_send_stat,
            self._cb_exit,
            self._cb_send_data,
            self._cb_send_init,
            self._cb_bg,
            None,
        )
        if rc != 0:
            raise RuntimeError(f"ngSpice_Init failed with code {rc}")

    def _command(self, command: str) -> None:
        if not command:
            return
        rc = self._lib.ngSpice_Command(command.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"ngSpice_Command('{command}') returned {rc}")

    def _load_circuit(self, netlist: str) -> None:
        lines: list[bytes | None] = []
        for raw_line in netlist.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lines.append(line.encode("utf-8"))
        if not any(line and line.lower().startswith(b".end") for line in lines if line):
            lines.append(b".end")
        lines.append(None)
        array_type = ctypes.c_char_p * len(lines)
        c_lines = array_type(*lines)
        rc = self._lib.ngSpice_Circ(c_lines)
        if rc != 0:
            raise RuntimeError("ngSpice_Circ failed to load circuit")

    # Callback bridges ---------------------------------------------------

    def _send_char(self, message: bytes, _id: int, _user: ctypes.c_void_p) -> int:
        if message:
            text = message.decode("utf-8", errors="ignore").strip()
            if text:
                self._logs.append(text)
                cb = None
                if self._active_callbacks:
                    cb = self._active_callbacks.on_message
                if cb is not None:
                    try:
                        cb(text)
                    except Exception:
                        pass
        return 0

    def _controlled_exit(
        self,
        status: int,
        immediate: bool,
        _exit_upon_quit: bool,
        _user: ctypes.c_void_p,
    ) -> int:
        # ngspice intends to exit the host process; we translate it into a log entry instead.
        self._logs.append(f"ngspice exit status={status} immediate={immediate}")
        return 0

    def _send_data(self, *args: object) -> int:
        if not self._external_sources:
            return 0
        if not args:
            return 0
        raw = args[0]
        self._logs.append("callback: send_data")
        try:
            ptr = ctypes.cast(cast(Any, raw), ctypes.POINTER(VecValuesAll))
        except Exception:
            return 0
        try:
            vec_all = ptr.contents
        except Exception:
            return 0

        index = int(vec_all.vecindex)
        values: dict[str, float] = {}
        time_value = 0.0
        if vec_all.veccount and vec_all.vecsa:
            for i in range(vec_all.veccount):
                ptr = vec_all.vecsa[i]
                if not ptr:
                    continue
                vec = ptr.contents
                name_bytes = vec.name or b""
                name = name_bytes.decode("utf-8", errors="ignore") if name_bytes else f"vec{i}"
                if vec.is_complex:
                    continue
                value = vec.real_value()
                values[name] = value
                if name.lower() == "time":
                    time_value = value

        if not values:
            return 0

        if len(self._last_source_values) == 0 and values:
            preview = {k: values[k] for k in list(values.keys())[:4]}
            self._logs.append(f"vecindex={index} preview={preview}")

        for src_name, provider in self._external_sources.items():
            try:
                result = float(
                    provider(ExternalSourceRequest(name=src_name, time=time_value, values=values))
                )
            except Exception:
                continue
            prev = self._last_source_values.get(src_name)
            if prev is not None and abs(prev - result) < 1e-12:
                continue
            self._last_source_values[src_name] = result
            commands = [
                f"alter @{src_name}[dc]={result}",
                f"alter {src_name} = {result}",
                f"alter {src_name} {result}",
            ]
            success = False
            for cmd in commands:
                try:
                    self._command(cmd)
                    self._logs.append(f"command: {cmd}")
                    success = True
                    break
                except Exception:
                    continue
            if not success:
                self._logs.append(f"command-failed: {src_name} -> {result}")

        return 0

    def _send_init_data(self, *_args: object) -> int:  # pragma: no cover - stub
        return 0

    def _bg_thread(self, _running: bool, _user: ctypes.c_void_p) -> int:  # pragma: no cover - stub
        return 0


def _resolve_library_path() -> str:
    attempted: list[str] = []
    for cand in _candidate_paths():
        if not cand:
            continue
        try:
            ctypes.CDLL(cand)
        except OSError:
            attempted.append(str(cand))
            continue
        else:
            return cand
    hints = [
        "Install ngspice with shared-library support (e.g. `brew install ngspice`).",
        "Set SPICELAB_NGSPICE_SHARED to the absolute path of libngspice (libngspice.dylib/.so).",
    ]
    raise EngineSharedLibraryNotFound("ngspice-shared", hints=hints, attempted=attempted)


def load_default_backend() -> CtypesNgSpiceSharedBackend:
    path = _resolve_library_path()
    return CtypesNgSpiceSharedBackend(path)
