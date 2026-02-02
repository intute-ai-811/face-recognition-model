# app/logging_config.py
import logging, sys, os, time
from contextlib import contextmanager
from datetime import datetime

LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41m",
}
RESET = "\033[0m"

STANDARD_ATTRS = {
    "name","msg","args","levelname","levelno","pathname","filename","module",
    "exc_info","exc_text","stack_info","lineno","funcName","created","msecs",
    "relativeCreated","thread","threadName","processName","process"
}

class KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = record.levelname

        use_color = sys.stdout.isatty() and os.getenv("LOG_COLOR", "1") == "1"
        color = LEVEL_COLORS.get(level, "") if use_color else ""
        reset = RESET if use_color else ""
        lvl = f"{color}{level:<8}{reset}"

        base = {
            "ts": ts,
            "level": lvl,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Capture python-logging "extra" fields correctly
        extras = {k: v for k, v in record.__dict__.items() if k not in STANDARD_ATTRS}
        base.update(extras)

        head = f"{base['ts']} | {base['level']} | {base['logger']}"
        tail = " | ".join(
            f"{k}={v}" for k, v in base.items()
            if k not in {"ts", "level", "logger"}
        )

        out = f"{head} | {tail}" if tail else head

        if record.exc_info:
            out += "\n" + self.formatException(record.exc_info)
        return out

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(KVFormatter())

def setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # safer than handlers.clear()
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)

    logging.root.setLevel(level)
    logging.root.addHandler(_handler)

    for noisy in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
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
        lvl = os.getenv("LOG_TIMED_LEVEL", "INFO").lower()
        fn = getattr(log, lvl, log.info)
        fn(event, extra={**fields, "ms": dt_ms})
