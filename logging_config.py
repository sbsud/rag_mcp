# logging_config.py
"""
Central logging setup for the entire project.

Call setup_logging() once at the entry point of each runnable file
(mcp_server_http.py, mcp_server_websearch.py, multi_agent.py, ingest.py).

Every other module just does:
    import logging
    logger = logging.getLogger(__name__)

That gives each module its own named logger that inherits the root
configuration set here. No module needs to know about handlers or
formatters — that is all decided in one place.

Log levels used consistently across the project:
  DEBUG   — internal values: vectors, chunk text, raw API payloads
  INFO    — key events: tool called, step started, file ingested
  WARNING — degraded but recoverable: empty result, truncated content
  ERROR   — failed operation with exception
"""

import logging
import logging.handlers
import sys
from pathlib import Path
import config


# ── ANSI colour codes for console output ──────────────────────────────────
# Only applied to the console handler, not the log file.

RESET  = "\x1b[0m"
GREY   = "\x1b[38;5;240m"
CYAN   = "\x1b[36m"
GREEN  = "\x1b[32m"
YELLOW = "\x1b[33m"
RED    = "\x1b[31m"
BOLD   = "\x1b[1m"

LEVEL_COLOURS = {
    logging.DEBUG:    GREY,
    logging.INFO:     GREEN,
    logging.WARNING:  YELLOW,
    logging.ERROR:    RED,
    logging.CRITICAL: BOLD + RED,
}


class ColourFormatter(logging.Formatter):
    """
    Formatter for the console handler.
    Colours the level name; keeps the rest readable.

    Format:
      HH:MM:SS.mmm  INFO     embedder:42          Embedded 5 texts in 0.31s
    """

    FMT = "{time}  {level:<8} {location:<28} {message}"

    def format(self, record: logging.LogRecord) -> str:
        colour = LEVEL_COLOURS.get(record.levelno, RESET)
        time   = self.formatTime(record, "%H:%M:%S") + f".{record.msecs:03.0f}"
        level  = f"{colour}{record.levelname}{RESET}"
        loc    = f"{record.name}:{record.lineno}"
        msg    = record.getMessage()

        # Attach exception info if present
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return self.FMT.format(time=time, level=level, location=loc, message=msg)


class PlainFormatter(logging.Formatter):
    """
    Formatter for the rotating file handler.
    No colour codes — clean text for grep and tail -f.

    Format:
      2025-05-29 14:32:01.042  INFO     embedder:42  Embedded 5 texts in 0.31s
    """
    def format(self, record: logging.LogRecord) -> str:
        time = self.formatTime(record, "%Y-%m-%d %H:%M:%S") + f".{record.msecs:03.0f}"
        loc  = f"{record.name}:{record.lineno}"
        msg  = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return f"{time}  {record.levelname:<8} {loc:<32} {msg}"


def setup_logging() -> None:
    """
    Configure the root logger once.
    Safe to call multiple times — subsequent calls are no-ops.
    """
    root = logging.getLogger()

    # Guard: already configured
    if root.handlers:
        return

    root.setLevel(config.LOG_LEVEL_NAME)

    # ── Console handler ────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(config.LOG_LEVEL_NAME)
    console.setFormatter(ColourFormatter())
    root.addHandler(console)

    # ── Rotating file handler ──────────────────────────────────────────────
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename    = log_dir / "rag_mcp.log",
        maxBytes    = 5 * 1024 * 1024,   # 5 MB per file
        backupCount = 3,                  # keep last 3 rotated files
        encoding    = "utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # file always gets DEBUG
    file_handler.setFormatter(PlainFormatter())
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    root.info("Logging initialised — level=%s  file=%s/rag_mcp.log",
              config.LOG_LEVEL_NAME, config.LOG_DIR)