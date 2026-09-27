"""Build a metadata CSV for LibriSpeech.

Walks the LibriSpeech directory tree and produces a CSV with columns:
    signal_path  — path relative to --root (resolved at training time)
    corpus       — fixed value matching the dataset_roots key in the config
    split        — train / val / test

Subset → split mapping
----------------------
  train-clean-100, train-clean-360, train-other-500  →  train
  dev-clean, dev-other                               →  val
  test-clean, test-other                             →  test

Unknown subsets are skipped with a warning.

Usage
-----
    python scripts/build_librispeech_metadata.py \\
        --root /path/to/LibriSpeech \\
        --output /path/to/librispeech_metadata.csv \\
        [--corpus librispeech]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

SUBSET_TO_SPLIT: dict[str, str] = {
    "train-clean-100": "train",
    "train-clean-360": "train",
    "train-other-500": "train",
    "dev-clean":       "val",
    "dev-other":       "val",
    "test-clean":      "test",
    "test-other":      "test",
}


def build_metadata(root: Path, corpus: str) -> pd.DataFrame:
    rows: list[dict] = []
    known_subsets = set(SUBSET_TO_SPLIT)
    seen_unknown: set[str] = set()

    for flac in sorted(root.rglob("*.flac")):
        try:
            rel = flac.relative_to(root)
        except ValueError:
            continue

        subset = rel.parts[0]
        if subset not in known_subsets:
            if subset not in seen_unknown:
                warnings.warn(f"Unknown subset {subset!r} — skipping", stacklevel=2)
                seen_unknown.add(subset)
            continue

        rows.append({
            "signal_path": str(rel),
            "corpus": corpus,
            "split": SUBSET_TO_SPLIT[subset],
        })

    return pd.DataFrame(rows, columns=["signal_path", "corpus", "split"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",   required=True,  type=Path, help="LibriSpeech root directory.")
    parser.add_argument("--output", required=True,  type=Path, help="Output CSV path.")
    parser.add_argument("--corpus", default="librispeech",
                        help="Corpus name (must match dataset_roots key in config).")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"Root directory not found: {args.root}")

    print(f"Scanning {args.root} ...")
    df = build_metadata(args.root, args.corpus)

    counts = df["split"].value_counts().to_dict()
    print(f"Found {len(df)} files:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {counts.get(split, 0)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
