import logging
import logging.handlers
from pathlib import Path
from datetime import datetime, timezone

KRISP_DIR = Path.home() / ".krisp"
LOG_DIR = KRISP_DIR / "logs"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

_initialized = False  # module-level guard


def get_logger(name: str) -> logging.Logger:
    _ensure_logging()
    return logging.getLogger(f"krisp.{name}")


def setup_logging(log_to_terminal: bool = False, verbose: bool = False) -> None:
    """
    Called explicitly by the watcher entrypoint.
    Safe to call multiple times — does nothing after first call.
    """
    global _initialized
    if _initialized:
        return
    _initialize(log_to_terminal=log_to_terminal, verbose=verbose)


def _ensure_logging() -> None:
    """
    Called internally by get_logger().
    Guarantees logging is set up even when no entrypoint called setup_logging().
    """
    global _initialized
    if _initialized:
        return
    _initialize(log_to_terminal=False, verbose=False)


def _initialize(log_to_terminal: bool, verbose: bool) -> None:
    global _initialized

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_filename = LOG_DIR / f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%S')}.log"

    root = logging.getLogger("krisp")
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    fh = logging.handlers.RotatingFileHandler(
        log_filename,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    if log_to_terminal:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    _initialized = True
