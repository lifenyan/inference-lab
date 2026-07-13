"""Structured logging helpers.

Human-readable logs go to stderr via ``get_logger``; machine-readable event
records (per-request measurements, experiment metadata) are appended as JSON
lines via ``log_event`` so downstream analysis never parses log text.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a stderr logger namespaced under ``inference_lab``."""
    logger = logging.getLogger(f"inference_lab.{name}")
    if not logging.getLogger("inference_lab").handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root = logging.getLogger("inference_lab")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return logger


def log_event(path: Path, event: dict[str, Any]) -> None:
    """Append one JSON event record to ``path``, adding a unix ``ts`` if absent."""
    event.setdefault("ts", time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
