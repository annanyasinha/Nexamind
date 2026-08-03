import logging
import sys
from typing import Optional


def setup_logger(name: str = "rag_platform", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a structured logger for the RAG platform.
    """
    logger_inst = logging.getLogger(name)
    if not logger_inst.handlers:
        logger_inst.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger_inst.addHandler(handler)
        logger_inst.propagate = False
    return logger_inst


logger = setup_logger()
