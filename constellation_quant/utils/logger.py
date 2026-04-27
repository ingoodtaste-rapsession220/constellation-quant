"""Centralized logging configuration.

Colored console output + optional JSON-structured file logs. All modules should
import `get_logger(__name__)` rather than configuring their own loggers.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from loguru import logger as _loguru_logger
    _HAS_LOGURU = True
except ImportError:
    _HAS_LOGURU = False


_CONFIGURED = False


def _configure_loguru(log_dir: Optional[Path], level: str) -> None:
    """Configure loguru with console + file sinks."""
    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            log_dir / "constellation_quant_{time:YYYY-MM-DD}.log",
            level=level,
            serialize=True,
            rotation="100 MB",
            retention="30 days",
        )


def _configure_stdlib(log_dir: Optional[Path], level: str) -> None:
    """Fallback stdlib logging when loguru is unavailable."""
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "constellation_quant.log")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def configure(log_dir: Optional[Path] = None, level: Optional[str] = None) -> None:
    """Configure logging globally. Safe to call multiple times."""
    global _CONFIGURED
    resolved_level = (level or os.environ.get("DYNAGRAPH_LOG_LEVEL", "INFO")).upper()

    if _HAS_LOGURU:
        _configure_loguru(log_dir, resolved_level)
    else:
        _configure_stdlib(log_dir, resolved_level)
    _CONFIGURED = True


def get_logger(name: str = "constellation_quant"):
    """Return a logger. Auto-configures on first call."""
    if not _CONFIGURED:
        configure()
    if _HAS_LOGURU:
        return _loguru_logger.bind(name=name)
    return logging.getLogger(name)
