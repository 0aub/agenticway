from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

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

    start_time = time.time()

    for file_path in cfg.files:
        process_pdf(Path(file_path).expanduser(), cfg)

    total_time = time.time() - start_time  
    elapsed_time = datetime.timedelta(seconds=int(total_time))  
    print(f"\n\n✅  Total experiment time: {elapsed_time}") 


if __name__ == "__main__":
    _cli()
