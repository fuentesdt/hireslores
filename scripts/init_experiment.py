#!/usr/bin/env python3
"""Copy analysis artifacts from a base results dir into a new experiment dir.

mist_train requires config.json, train_paths.csv, and fg_bboxes.csv to already
exist in --results.  Each (dataset, loss) experiment needs its own isolated
copy because mist_train overwrites config.json with the CLI-specified loss.

Usage:
    python3 scripts/init_experiment.py results/hires_base results/hires_dice_ce
"""
import sys
import shutil
from pathlib import Path

REQUIRED = ["config.json", "train_paths.csv", "fg_bboxes.csv"]
OPTIONAL = ["test_paths.csv"]


def main() -> None:
    base_dir = Path(sys.argv[1])
    exp_dir = Path(sys.argv[2])
    exp_dir.mkdir(parents=True, exist_ok=True)

    for fname in REQUIRED:
        src = base_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Required artifact missing: {src}")
        shutil.copy2(src, exp_dir / fname)

    for fname in OPTIONAL:
        src = base_dir / fname
        if src.exists():
            shutil.copy2(src, exp_dir / fname)

    print(f"Initialized {exp_dir} from {base_dir}")


if __name__ == "__main__":
    main()
