"""Build a metadata CSV from a directory of audio files.

Output columns: signal_path (relative to --root), corpus, split.

With explicit split directories:
    python scripts/build_metadata.py \\
        --root /data/my_dataset \\
        --output metadata.csv \\
        --corpus my_dataset \\
        --train-dirs train \\
        --val-dirs   dev \\
        --test-dirs  test

Without split directories (everything assigned to train):
    python scripts/build_metadata.py \\
        --root /data/my_dataset \\
        --output metadata.csv \\
        --corpus my_dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def walk_audio(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",       type=Path, required=True,
                        help="Dataset root directory.")
    parser.add_argument("--output",     type=Path, required=True,
                        help="Output CSV path.")
    parser.add_argument("--corpus",     type=str,  default="dataset",
                        help="Corpus name (must match dataset_roots key in config).")
    parser.add_argument("--train-dirs", nargs="*", default=[],
                        help="Root-relative subdirectory names for the train split.")
    parser.add_argument("--val-dirs",   nargs="*", default=[],
                        help="Root-relative subdirectory names for the val split.")
    parser.add_argument("--test-dirs",  nargs="*", default=[],
                        help="Root-relative subdirectory names for the test split.")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"Root directory not found: {args.root}")

    split_map = (
        [(d, "train") for d in args.train_dirs]
        + [(d, "val")  for d in args.val_dirs]
        + [(d, "test") for d in args.test_dirs]
    )

    rows = []
    if not split_map:
        for path in walk_audio(args.root):
            rows.append({"signal_path": str(path.relative_to(args.root)),
                         "corpus": args.corpus, "split": "train"})
    else:
        for subdir, split in split_map:
            sub = args.root / subdir
            if not sub.exists():
                print(f"Warning: {sub} does not exist, skipping.")
                continue
            for path in walk_audio(sub):
                rows.append({"signal_path": str(path.relative_to(args.root)),
                             "corpus": args.corpus, "split": split})

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    for split in ["train", "val", "test"]:
        n = (df["split"] == split).sum()
        if n:
            print(f"  {split}: {n} files")


if __name__ == "__main__":
    main()
