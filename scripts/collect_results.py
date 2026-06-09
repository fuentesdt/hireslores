#!/usr/bin/env python3
"""Aggregate cross-validation results across experiments into a Markdown table.

Reads results.csv from each experiment directory, computes mean ± std for
every metric column, and writes a comparison table to the output path.

Usage:
    python3 scripts/collect_results.py results/comparison.md \
        results/hires_dice_ce/results.csv \
        results/hires_cldice/results.csv \
        results/lores_dice_ce/results.csv \
        results/lores_cldice/results.csv
"""
import sys
import pandas as pd
from pathlib import Path


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(str(df.loc[i, c]) for c in cols) + " |"
        for i in df.index
    ]
    return "\n".join([header, sep] + rows)

NON_METRIC = {"id", "mask", "prediction", "fold"}


def summarize(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    metric_cols = [c for c in df.columns if c not in NON_METRIC]
    row: dict = {}
    for col in metric_cols:
        vals = df[col].dropna()
        row[col] = f"{vals.mean():.4f} ± {vals.std():.4f}"
    return row


def main() -> None:
    output_path = Path(sys.argv[1])
    csv_paths = [Path(p) for p in sys.argv[2:]]

    rows = []
    for csv_path in csv_paths:
        # Derive resolution and loss from parent directory name, e.g. hires_dice_ce
        name = csv_path.parent.name
        resolution, _, loss = name.partition("_")
        summary = summarize(csv_path)
        rows.append({"Resolution": resolution, "Loss": loss, **summary})

    df = pd.DataFrame(rows).sort_values(["Resolution", "Loss"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# HiRes vs LoRes — Loss Function Comparison\n\n")
        f.write(to_markdown(df))
        f.write("\n\n_Metrics are mean ± std across all cross-validation cases._\n")

    print(to_markdown(df))
    print(f"\nTable written to {output_path}")


if __name__ == "__main__":
    main()
