"""Download standard time series benchmark datasets from Autoformer repository."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"

DATASETS: dict[str, str] = {
    "ETTh1": f"{ETT_BASE}/ETTh1.csv",
    "ETTh2": f"{ETT_BASE}/ETTh2.csv",
    "ETTm1": f"{ETT_BASE}/ETTm1.csv",
    "ETTm2": f"{ETT_BASE}/ETTm2.csv",
}

LARGE_DATASETS: dict[str, str] = {
    "Weather": (
        "https://raw.githubusercontent.com/ts-kim/RevIN/master/datasets/weather/weather.csv"
    ),
    "Electricity": (
        "https://raw.githubusercontent.com/ts-kim/RevIN/master/datasets/electricity/electricity.csv"
    ),
    "Traffic": (
        "https://raw.githubusercontent.com/ts-kim/RevIN/master/datasets/traffic/traffic.csv"
    ),
}


def download_file(url: str, dest: Path) -> None:
    """Download a file if it doesn't already exist."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)
    print(f"  Saved to: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Specific datasets to download (default: all ETT)",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Also download Weather and Electricity (larger files)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    targets = DATASETS.copy()
    if args.include_large:
        targets.update(LARGE_DATASETS)

    if args.datasets:
        targets = {k: v for k, v in targets.items() if k in args.datasets}

    if not targets:
        print("No matching datasets found.")
        return

    for name, url in targets.items():
        print(f"\n[{name}]")
        ext = ".csv" if url.endswith(".csv") else ".txt"
        download_file(url, data_dir / f"{name}{ext}")

    print(f"\nDone. {len(targets)} dataset(s) in {data_dir}/")


if __name__ == "__main__":
    main()
