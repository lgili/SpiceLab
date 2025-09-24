from collections.abc import Sequence

import numpy as np
import pytest
from spicelab.analysis.core import AnalysisResult
from spicelab.analysis.post import stack_runs_to_df
from spicelab.io.raw_reader import Trace, TraceSet


def _make_res(xname: str, yname: str, xs: Sequence[float], ys: Sequence[float]) -> AnalysisResult:
    ts = TraceSet([Trace(xname, None, np.asarray(xs)), Trace(yname, None, np.asarray(ys))])
    # Dummy run result minimal
    from spicelab.spice.base import RunArtifacts, RunResult

    art = RunArtifacts(netlist_path="n", log_path="l", raw_path=None)
    rr = RunResult(artifacts=art, returncode=0, stdout="", stderr="")
    return AnalysisResult(run=rr, traces=ts)


def test_stack_runs_to_df_pick_columns_and_params() -> None:
    res1 = _make_res("time", "v(n1)", [0.0, 1.0], [1.0, 2.0])
    res2 = _make_res("time", "v(n1)", [0.0, 1.0], [3.0, 4.0])
    df = stack_runs_to_df([res1, res2], params_list=[{"P": 1}, {"P": 2}], y=["v(n1)"], with_x=True)
    assert list(df.columns) == ["time", "v(n1)", "P", "run_idx"]
    assert set(df["P"].unique()) == {1, 2}


def test_stack_runs_missing_column_raises() -> None:
    res = _make_res("time", "v(n1)", [0.0], [1.0])
    with pytest.raises(KeyError):
        _ = stack_runs_to_df([res], params_list=None, y=["v(out)"], with_x=False)
