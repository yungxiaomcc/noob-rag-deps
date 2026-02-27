"""
可配置、非阻塞日志：QueueHandler + QueueListener，本地默认写文件，支持 OTLP。
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import queue
import sys
import structlog
from structlog.stdlib import (
    BoundLogger,
    LoggerFactory,
    ProcessorFormatter,
    PositionalArgumentsFormatter,
    add_log_level,
    add_logger_name,
    filter_by_level,
)

from noob_rag_deps.config import LogConfig

_LISTENER: logging.handlers.QueueListener | None = None
_QUEUE: queue.Queue[logging.LogRecord] | None = None


def _make_processors() -> list[structlog.typing.Processor]:
    """structlog 处理器链：先合并 contextvars，再构建 event_dict，最后由 ProcessorFormatter 渲染。"""
    shared = [
        structlog.contextvars.merge_contextvars,
        filter_by_level,
        add_logger_name,
        add_log_level,
        PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]
    return shared


def _make_formatter(config: LogConfig) -> ProcessorFormatter:
    """根据 config.format 返回 ProcessorFormatter（json 或 text）。"""
    foreign_pre_chain = [
        add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if config.format == "json":
        renderer: structlog.typing.Processor = structlog.processors.JSONRenderer(
            indent=config.json_indent
        )
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)
    return ProcessorFormatter(
        foreign_pre_chain=foreign_pre_chain,
        processors=[
            ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )


def _create_otlp_handler(config: LogConfig) -> logging.Handler | None:
    """可选：OTLP 日志 Handler。未安装 opentelemetry 或未配置 endpoint 时返回 None。"""
    if not config.otlp_endpoint:
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        return None
    resource = Resource.create({"service.name": config.service_name})
    provider = LoggerProvider(resource=resource)
    exporter = OTLPLogExporter(endpoint=config.otlp_endpoint.rstrip("/"))
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    return LoggingHandler(level=logging.NOTSET, logger_provider=provider)


def setup_logging(config: LogConfig) -> None:
    """配置非阻塞日志：QueueHandler + QueueListener，stdout/文件/可选 OTLP。"""
    global _LISTENER, _QUEUE

    if _LISTENER is not None:
        shutdown_logging()

    structlog.configure(
        processors=_make_processors(),
        wrapper_class=BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = _make_formatter(config)
    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    handlers.append(stream_handler)

    if config.file_path and config.file_path.strip():
        path = config.file_path.strip()
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=config.file_max_bytes,
            backupCount=config.file_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, config.level.upper(), logging.INFO))
        handlers.append(file_handler)

    otlp_handler = _create_otlp_handler(config)
    if otlp_handler is not None:
        otlp_handler.setLevel(getattr(logging, config.level.upper(), logging.INFO))
        handlers.append(otlp_handler)

    _QUEUE = queue.Queue(maxsize=config.queue_size)
    _LISTENER = logging.handlers.QueueListener(
        _QUEUE,
        *handlers,
        respect_handler_level=True,
    )
    _LISTENER.start()
    queue_handler = logging.handlers.QueueHandler(_QUEUE)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(queue_handler)
    root.setLevel(getattr(logging, config.level.upper(), logging.INFO))


def shutdown_logging() -> None:
    """应用退出时调用，排空队列并停止 QueueListener。"""
    global _LISTENER
    if _LISTENER is not None:
        _LISTENER.stop()
        _LISTENER = None


def get_logger(name: str = "noob_rag_deps") -> BoundLogger:
    """返回 structlog 绑定的 logger，支持 contextvars 上下文。"""
    return structlog.get_logger(name)  # type: ignore[return-value]
