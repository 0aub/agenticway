from __future__ import annotations
import argparse
import sys
from pathlib import Path

from config import AppConfig, ConfigError
from pipeline import process_pdf


def _cli() -> None:
    pa = argparse.ArgumentParser(prog="slide2vid", description="Slides ➜ dialog ➜ narrated MP4")
    pa.add_argument("-c", "--config", default='config.yaml', type=Path, help="Path to config.yaml")
    args = pa.parse_args()

    try:
        cfg = AppConfig.from_yaml(args.config)
    except ConfigError as e:
        sys.exit(f"❌ {e}")

    if cfg.test_mode and cfg.files:
        cfg.files = cfg.files[:1]

    for pdf_path in cfg.files:
        process_pdf(Path(pdf_path).expanduser(), cfg)


if __name__ == "__main__":
    _cli()
