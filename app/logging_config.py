# app/logging_config.py
import logging, sys, os, time
from contextlib import contextmanager
from datetime import datetime

LEVEL_COLORS = {
    "DEBUG": "\033[36m",   # Cyan
    "INFO": "\033[32m",    # Green
    "WARNING": "\033[33m", # Yellow
    "ERROR": "\033[31m",   # Red
    "CRITICAL": "\033[41m" # Red background
}
RESET = "\033[0m"

class KVFormatter(logging.Formatter):
    def format(self, record):
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = record.levelname
        color = LEVEL_COLORS.get(level, "")
        lvl = f"{color}{level:<8}{RESET}"

        base = {
            "ts": ts,
            "level": lvl,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        if hasattr(record, "extra") and isinstance(record.extra, dict):
            base.update(record.extra)

        head = f"{base['ts']} | {base['level']} | {base['logger']}"
        tail = " | ".join(f"{k}={v}" for k, v in base.items() if k not in {"ts", "level", "logger"})

        return f"{head} | {tail}" if tail else head

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(KVFormatter())

def setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.root.handlers.clear()
    logging.root.setLevel(level)
    logging.root.addHandler(_handler)

    for noisy in ["uvicorn", "uvicorn.error"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

@contextmanager
def timed(log: logging.Logger, event: str, **fields):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        log.info(event, extra={**fields, "ms": dt_ms})
