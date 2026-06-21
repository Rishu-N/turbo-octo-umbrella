"""Download and cache public datasets used for benchmarking, OOS calibration,
and the Hindi/Hinglish tail.

This is the ONLY build-time network entry point in the project. The inference
hot path (Phases 1+) never calls out.

Datasets (defaults):
  - PolyAI/banking77            (benchmark, EN, CC-BY-4.0)
  - clinc_oos (config "plus")   (OOS calibration, EN, CC-BY-SA 3.0 — eval only)
  - AmazonScience/massive hi-IN (Hindi tail, CC-BY-4.0)

Optional:
  - HWU64 via --include-hwu64   (extra eval, CC-BY-SA 3.0 — eval only)

Idempotent: skips datasets already present on disk unless --force is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


DEFAULT_DATASETS: list[dict] = [
    {
        "name": "banking77",
        "loader": ("PolyAI/banking77", None),
        "note": "77 fine-grained banking intents, English",
    },
    {
        "name": "clinc_oos_plus",
        "loader": ("clinc_oos", "plus"),
        "note": "150 intents + out-of-scope examples (OOS calibration)",
    },
    {
        "name": "massive_hi_IN",
        "loader": ("AmazonScience/massive", "hi-IN"),
        "note": "60 intents, Hindi locale",
    },
]

OPTIONAL_DATASETS: list[dict] = [
    {
        "name": "hwu64",
        "loader": ("dialoglue/hwu64", None),
        "note": "64 multi-domain intents (eval only)",
    },
]


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def dataset_dir(data_raw: Path, name: str) -> Path:
    return data_raw / name


def download_one(spec: dict, data_raw: Path, force: bool) -> tuple[str, dict[str, int] | str]:
    name = spec["name"]
    repo, config = spec["loader"]
    out_dir = dataset_dir(data_raw, name)

    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        # Already present — count rows from the cached copy for the summary table.
        try:
            from datasets import load_from_disk

            ds = load_from_disk(str(out_dir))
            splits = {split: len(ds[split]) for split in ds} if hasattr(ds, "keys") else {"all": len(ds)}
            return name, splits
        except Exception as exc:  # noqa: BLE001
            return name, f"cached (could not introspect: {exc})"

    print(f"[download] {name}: fetching {repo}" + (f" (config={config})" if config else ""))
    ds = load_dataset(repo, config) if config else load_dataset(repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))

    splits = {split: len(ds[split]) for split in ds} if hasattr(ds, "keys") else {"all": len(ds)}
    return name, splits


def print_summary(rows: list[tuple[str, dict[str, int] | str]]) -> None:
    print("\nDataset summary:")
    print(f"  {'name':<22}  {'splits / counts'}")
    print(f"  {'-' * 22}  {'-' * 40}")
    for name, info in rows:
        if isinstance(info, dict):
            counts = ", ".join(f"{k}={v}" for k, v in info.items())
        else:
            counts = info
        print(f"  {name:<22}  {counts}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download public datasets to data/raw/")
    parser.add_argument("--include-hwu64", action="store_true", help="Also download HWU64 (optional)")
    parser.add_argument("--force", action="store_true", help="Re-download even if already cached")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data/raw directory (defaults to config paths.data_raw)",
    )
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    data_raw = args.data_dir or (REPO_ROOT / cfg["paths"]["data_raw"])
    data_raw.mkdir(parents=True, exist_ok=True)

    targets = list(DEFAULT_DATASETS)
    if args.include_hwu64:
        targets.extend(OPTIONAL_DATASETS)

    print(f"Cache directory: {data_raw}")

    results: list[tuple[str, dict[str, int] | str]] = []
    for spec in targets:
        try:
            results.append(download_one(spec, data_raw, args.force))
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {spec['name']}: {exc}", file=sys.stderr)
            results.append((spec["name"], f"FAILED: {exc}"))

    print_summary(results)

    failures = [name for name, info in results if isinstance(info, str) and info.startswith("FAILED")]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
