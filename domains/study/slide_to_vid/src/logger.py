from __future__ import annotations
import logging
from pathlib import Path

try:
    from rich.logging import RichHandler

    _HAVE_RICH = True
except ModuleNotFoundError:
    _HAVE_RICH = False


def init_logger(run_dir: Path, *, use_rich: bool = True) -> logging.Logger:
    """
    Coloured console + file logging.

    A `run.log` file is created inside *run_dir*.
    If Rich isn't installed or `use_rich=False`, falls back to std logging.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    console_fmt = "%(message)s"
    file_fmt = "%(asctime)s — %(levelname)s — %(name)s — %(message)s"

    handlers: list[logging.Handler] = []

    # console
    if use_rich and _HAVE_RICH:
        handlers.append(RichHandler(markup=True, rich_tracebacks=True))
    else:
        handlers.append(logging.StreamHandler())

    # file
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(file_fmt))
    handlers.append(file_handler)

    logging.basicConfig(level=logging.INFO, format=console_fmt, handlers=handlers)

    return logging.getLogger("slide_to_vid")
