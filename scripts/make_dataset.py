#!/usr/bin/env python3
"""Create MIST raw/train directory structure with symlinks to original NIfTI files.

Reads a training CSV (id, mask, image, ...) and creates symlinks under
<output_dir>/raw/train/<id>/ so that mist_analyze can find the data without
copying the large NIfTI files.

Usage:
    python3 scripts/make_dataset.py data/hires.csv datasets/hires
"""
import sys
import pandas as pd
from pathlib import Path


def main() -> None:
    csv_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    df = pd.read_csv(csv_path)
    train_dir = output_dir / "raw" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Columns: id, mask, then one or more image columns
    image_cols = list(df.columns)[2:]

    for _, row in df.iterrows():
        patient_dir = train_dir / str(row["id"])
        patient_dir.mkdir(exist_ok=True)

        mask_link = patient_dir / "mask.nii.gz"
        if not mask_link.exists():
            mask_link.symlink_to(Path(row["mask"]).resolve())

        for col in image_cols:
            img_link = patient_dir / f"{col}.nii.gz"
            if not img_link.exists():
                img_link.symlink_to(Path(row[col]).resolve())

    print(f"Created {len(df)} patient directories under {train_dir}")


if __name__ == "__main__":
    main()
