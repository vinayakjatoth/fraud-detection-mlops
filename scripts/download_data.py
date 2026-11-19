"""Download the credit-card fraud dataset to data/raw/creditcard.csv.

python scripts/download_data.py                  # OpenML (no account needed)
python scripts/download_data.py --source kaggle  # needs the Kaggle CLI + API token
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from fraud_detection.config import load_config
from fraud_detection.data import download_openml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--source", choices=["openml", "kaggle"], default="openml")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dest = Path(cfg["data"]["raw_path"])
    if dest.exists() and not args.force:
        print(f"{dest} already exists - nothing to do (use --force to re-download).")
        return

    if args.source == "openml":
        download_openml(cfg["data"]["openml_id"], dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "mlg-ulb/creditcardfraud",
                "-p",
                str(dest.parent),
                "--unzip",
            ],
            check=True,
        )
    print(f"Dataset ready at {dest}")


if __name__ == "__main__":
    sys.exit(main())
