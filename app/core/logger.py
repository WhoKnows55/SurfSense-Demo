"""
SurfSense Logging System

Consistent, structured logging across all components.
Supports both console and file output with configurable formats.
"""

import logging
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level

# Sensitive patterns to filter from logs
SENSITIVE_PATTERNS = ["api_key", "password", "secret", "token", "authorization"]


def filter_sensitive_data(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Remove sensitive data from log entries.

    Prevents accidental logging of API keys, passwords, and other secrets.
    """
    for key in list(event_dict.keys()):
        key_lower = key.lower()
        if any(pattern in key_lower for pattern in SENSITIVE_PATTERNS):
            event_dict[key] = "[REDACTED]"

    return event_dict


def add_app_context(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add SurfSense application context to log entries."""
    event_dict["app"] = "surfsense"
    return event_dict


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable console log formatter with colors.

    Format: [TIMESTAMP] LEVEL    | module | message
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with colors for console output."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        color = self.COLORS.get(level, "")

        # Pad level name for alignment
        level_padded = f"{level:<8}"

        # Format: [timestamp] LEVEL    | module | message
        formatted = (
            f"[{timestamp}] {color}{level_padded}{self.RESET} | "
            f"{record.name:<20} | {record.getMessage()}"
        )

        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Outputs each log entry as a single JSON line for easy parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        import json

        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "taskName", "message",
            ):
                # Filter sensitive data
                if any(p in key.lower() for p in SENSITIVE_PATTERNS):
                    log_data[key] = "[REDACTED]"
                else:
                    log_data[key] = value

        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    log_format: str = "text",
    log_file: Path | None = None,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ('json' or 'text')
        log_file: Path to log file (optional, logs to file if provided)
    """
    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if log_format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())

    root_logger.addHandler(console_handler)

    # File handler (if path provided)
    if log_file:
        # Ensure directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        # Always use JSON for file logs (easier to parse)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # Configure structlog for structured logging
    structlog.configure(
        processors=[
            filter_sensitive_data,
            add_app_context,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            add_log_level,
            TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            JSONRenderer() if log_format == "json" else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


@lru_cache
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing forecast", extra={"location": "Pipeline"})
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class that provides a logger attribute to any class.

    Usage:
        class MyService(LoggerMixin):
            def do_something(self):
                self.log_info("Doing something")
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__)
    
    def log_debug(self, msg: str, **kwargs) -> None:
        """Log a debug message."""
        self.logger.debug(msg, extra=kwargs if kwargs else None)
    
    def log_info(self, msg: str, **kwargs) -> None:
        """Log an info message."""
        self.logger.info(msg, extra=kwargs if kwargs else None)
    
    def log_warning(self, msg: str, **kwargs) -> None:
        """Log a warning message."""
        self.logger.warning(msg, extra=kwargs if kwargs else None)
    
    def log_error(self, msg: str, **kwargs) -> None:
        """Log an error message."""
        self.logger.error(msg, extra=kwargs if kwargs else None)


# Request/Response logging helpers for API calls
def log_api_request(
    logger: logging.Logger,
    method: str,
    url: str,
    params: dict[str, Any] | None = None,
) -> None:
    """
    Log an outgoing API request.

    Args:
        logger: Logger instance to use
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        params: Query parameters (sensitive data will be redacted)
    """
    # Redact sensitive params
    safe_params = {}
    if params:
        for key, value in params.items():
            if any(p in key.lower() for p in SENSITIVE_PATTERNS):
                safe_params[key] = "[REDACTED]"
            else:
                safe_params[key] = value

    logger.info(
        f"API Request: {method} {url}",
        extra={"method": method, "url": url, "params": safe_params},
    )


def log_api_response(
    logger: logging.Logger,
    status_code: int,
    response_time_ms: float,
    url: str,
) -> None:
    """
    Log an API response.

    Args:
        logger: Logger instance to use
        status_code: HTTP response status code
        response_time_ms: Response time in milliseconds
        url: Request URL
    """
    level = logging.INFO if status_code < 400 else logging.WARNING

    logger.log(
        level,
        f"API Response: {status_code} ({response_time_ms:.0f}ms) {url}",
        extra={
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "url": url,
        },
    )
