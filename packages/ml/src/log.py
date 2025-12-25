import logging
import sys
from ml.env import LOGS_FILE, LOG_LEVEL
from pathlib import Path
from datetime import datetime

_LOGGING_FLAG = "_ML_LOGGING_CONFIGURED"


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython
    except Exception:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    return "IPKernelApp" in ip.config

def _is_configured() -> bool:
    return bool(getattr(logging, _LOGGING_FLAG, False))

def _mark_configured() -> None:
    setattr(logging, _LOGGING_FLAG, True)


def configure_logging(logs_file: str = LOGS_FILE) -> logging.Logger:
    log_level = logging._nameToLevel.get(LOG_LEVEL.upper())
    Path(logs_file).parent.mkdir(parents=True, exist_ok=True)
    pure_name = Path(logs_file).stem
    extension = Path(logs_file).suffix
    pure_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_file = str(Path(logs_file).parent / (pure_name + extension))
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()
    fmt = logging.Formatter('[%(levelname)s] %(asctime)s: %(message)s')
    logger.addHandler(logging.FileHandler(logs_file, mode='w+'))
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    for handler in logger.handlers:
        handler.setFormatter(fmt)
    logger.info(f"Логирование настроено с уровнем: {LOG_LEVEL} ({log_level}).")
    _mark_configured()
    return logger


if _is_notebook() and not _is_configured():
    configure_logging()
