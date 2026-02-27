from noob_rag_deps.config import conf
from noob_rag_deps.logging import get_logger, setup_logging, shutdown_logging

setup_logging(conf.log)

__all__ = [
    "conf",
    "get_logger",
    "setup_logging",
    "shutdown_logging",
]
