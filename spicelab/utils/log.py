from __future__ import annotations

import logging
import os

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _env_level() -> int:
    lvl = os.getenv("CAT_LOG_LEVEL", "INFO").upper().strip()
    return _LEVELS.get(lvl, logging.INFO)


def get_logger(name: str = "spicelab") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(_env_level())
    return logger
