#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make 2x4 figure panels per PDB from af_pdb (e.g., '1ALB.A'):
Row 1 (bar):   [af3_bound, pdb_bound, af3_unbound, pdb_unbound]
Row 2 (scatter):
  [pnas_af3_lig, pdb_structures_final bound, pnas_af3_nolig, pdb_structures_final unbound]

Author: Maria L.
"""

import argparse
import sys
import re
from pathlib import Path
import polars as pl
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ---------- Helpers ----------

proteins = {
    "Bovine β-Lactoglobulin": "BLG",
    "KRAS": "K-RAS",
    "MAPK": "MAPK",
    "Pyruvate Dehydrogenase Kinase": "PDK",
    "Ribonuclease A": "RNase A",
    "β-Secretase": "BACE",
    "TEM β-lactamase": "TEM",
    "cAMP-dependent protein kinase ": "PKA",
    "Glutamate receptor 2": "GluR2",
    "AMPc Beta-Lactamase": "AmpC",
    "Thrombin": "Thrombin",
    "ALDBP": "ALDBP",
    "Myosin II": "Myosin 2",
    "Ricin": "Ricin",
    "Androgen receptor": "AR",
    "Hsp90": "Hsp90",
 }

def parse_pdb_id(cell: str) -> str:
    """
    '1ALB.A' -> '1ALB'
    """
    if not cell:
        return None
    return str(cell).split(".")[0].upper()


def read_counts_table(path: Path) -> pl.DataFrame:
    """
    Expecting a CSV with two columns:
      - State  (e.g., 'af3_bound_1ALB_closed')
      - counts (float)
    Returns a standardized dataframe with columns:
      - key
      - value
    """
    if not path.exists():
        raise FileNotFoundError(f"Counts CSV not found: {path}")

    df = pl.read_csv(path, infer_schema_length=10000)

    # Ensure required columns exist
    if not {"State", "Count"}.issubset(df.columns):
        raise ValueError(
            f"CSV must contain 'State' and 'Count' columns, found {df.columns}"
        )

    # Standardize column names
    df = df.select(
        [
            pl.col("State").cast(pl.Utf8).alias("key"),
            pl.col("Count").cast(pl.Float64).alias("value"),
        ]
    )
    return df


def counts_for(df_counts: pl.DataFrame, prefix: str, pdb: str):
    """
    Pull counts for closed/open/other given a prefix and pdb
    Example keys:
      af3_bound_1ALB_closed, af3_bound_1ALB_open, af3_bound_1ALB_other
    Returns [closed, open, other] (floats; 0 if missing)
    """
    raw_states = ["closed", "open", "neither"]  # what your CSV actually has
    display_states = ["closed", "open", "other"]  # what you want to show
    values = []
    for st in raw_states:
        key = f"{prefix}_{pdb}_{st}"
        v = df_counts.filter(pl.col("key") == key).select("value")
        if v.height == 0:
            values.append(0.0)
        else:
            values.append(float(v.item()))
    return display_states, values


def safe_read_loop_rmsd(csv_path: Path) -> np.ndarray:
    """
    Read 'loop_rmsd' column as np array. If missing file/column -> empty array.
    """
    if not csv_path.exists():
        return np.array([])
    try:
        df = pl.read_csv(csv_path, infer_schema_length=10000)
        if "loop_rmsd" in df.columns:
            vals = df.select(pl.col("loop_rmsd")).to_series().to_numpy()
            return np.array(vals, dtype=float)
        else:
            # try lowercase fallback
            lower_cols = {c.lower(): c for c in df.columns}
            if "loop_rmsd" in lower_cols:
                vals = df.select(pl.col(lower_cols["loop_rmsd"])).to_series().to_numpy()
                return np.array(vals, dtype=float)
            return np.array([])
    except Exception:
        return np.array([])


def annotate_missing(ax, msg: str):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def scatter_xy(ax, x: np.ndarray, y: np.ndarray, range: tuple):
    if x.size == 0 or y.size == 0:
        annotate_missing(ax, "missing data")
        return
    n = min(x.size, y.size)
    ax.scatter(x[:n], y[:n], s=30, alpha=0.5)
    ax.set_xlabel("RMSD to Closed Structure (Å)", fontsize=12)
    ax.set_ylabel("RMSD to Open Structure (Å)", fontsize=12)
    ax.set_xlim(-0.5, range[0])
    ax.set_ylim(-0.5, range[1])
    ax.grid(True, linestyle="--", alpha=0.6)


def bar_states(ax, states, values, title):
    colors = ["#76acd8", "#dfd1d1", "#746d72"]  # blue, gray, purple

    bars = ax.bar(
        states,
        values,
        color=colors[: len(states)],
        edgecolor="black",  # add bar outlines
        linewidth=1.2,
    )

    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Conformation", fontsize=12)
    max_val = max(values) if values else 1
    ax.set_ylim(0, max_val * 1.15)
    ax.set_title(f'{title}', fontsize=16)

    # add value labels on top of bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(values) * 0.02,  # small offset above bar
            f"{val:.0f}",  # integer style
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def make_figure_for_pdb(
    pdb_id: str,
    df_counts: pl.DataFrame,
    base_pnas_af3_lig: Path,
    base_pnas_af3_nolig: Path,
    base_pdb_structures: Path,
    out_dir: Path,
    range: tuple,
    series_name: str,
):
    """
    Build 2x4 grid for given pdb_id and save to out_dir/pdb_id.png
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=200)
    plt.subplots_adjust(wspace=0.30, hspace=0.35)

    # ---- Row 1: bar charts ----
    top_specs = [
        ("af3_bound", f"AF3 w/ Ligand"),
        ("pdb_bound", f"PDB Bound"),
        ("af3_unbound", f"AF3 w/o Ligand"),
        ("pdb_unbound", f"PDB Unbound"),
    ]
    for j, (prefix, title) in enumerate(top_specs):
        states, values = counts_for(df_counts, prefix, pdb_id)
        bar_states(axes[0, j], states, values, title)

    # ---- Row 2: scatter plots ----
    # (1) pnas_af3_lig/1alb/closed_rmsd.csv vs open_rmsd.csv
    lig_dir = base_pnas_af3_lig / pdb_id.lower()
    x1 = safe_read_loop_rmsd(lig_dir / "closed_rmsd.csv")
    y1 = safe_read_loop_rmsd(lig_dir / "open_rmsd.csv")
    scatter_xy(axes[1, 0], x1, y1, range)

    # (2) pdb_structures_final/1ALB_pdbs/bound/...
    bound_dir = base_pdb_structures / f"{pdb_id}_pdbs" / "bound"
    x2 = safe_read_loop_rmsd(bound_dir / "closed_rmsd.csv")
    y2 = safe_read_loop_rmsd(bound_dir / "open_rmsd.csv")
    scatter_xy(axes[1, 1], x2, y2, range)

    # (3) pnas_af3_nolig/1alb/...
    nolig_dir = base_pnas_af3_nolig / pdb_id.lower()
    x3 = safe_read_loop_rmsd(nolig_dir / "closed_rmsd.csv")
    y3 = safe_read_loop_rmsd(nolig_dir / "open_rmsd.csv")
    scatter_xy(axes[1, 2], x3, y3, range)

    # (4) pdb_structures_final/1ALB_pdbs/unbound/... (typo fallback: pdb_strictures_final)
    unbound_dir = base_pdb_structures / f"{pdb_id}_pdbs" / "unbound"
    x4 = safe_read_loop_rmsd(unbound_dir / "closed_rmsd.csv")
    y4 = safe_read_loop_rmsd(unbound_dir / "open_rmsd.csv")
    scatter_xy(axes[1, 3], x4, y4, range)

    plt.suptitle(f"{proteins.get(series_name)}", fontsize=20, y=0.95)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdb_id}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------- Main ----------


def main():
    df_csv = Path("pnas_table_mod.csv")
    counts_csv = Path("total_state_counts.csv")
    base_pnas_af3_lig = Path("pnas_af3_lig")
    base_pnas_af3_nolig = Path("pnas_af3_nolig")
    base_pdb_structures = Path("pdb_structures_final")
    out_dir = Path("plots_out")

    # Load inputs
    try:
        df = pl.read_csv(df_csv, infer_schema_length=10000)
    except Exception as e:
        print(f"[ERROR] Failed to read df_csv: {df_csv} -> {e}", file=sys.stderr)
        sys.exit(1)

    if "af_pdb" not in df.columns:
        # try relaxed match
        candidates = [c for c in df.columns if c.lower() == "af_pdb"]
        if not candidates:
            print(
                "[ERROR] Input dataframe must have an 'af_pdb' column.", file=sys.stderr
            )
            sys.exit(1)
        df = df.rename({candidates[0]: "af_pdb"})
    

    try:
        df_counts = read_counts_table(counts_csv)
    except Exception as e:
        print(
            f"[ERROR] Failed to read counts_csv: {counts_csv} -> {e}", file=sys.stderr
        )
        sys.exit(1)

    # Iterate rows
    series = df.select("af_pdb").to_series().to_list()
    series_min = df.select("min").to_series().to_list()
    series_max = df.select("max").to_series().to_list()
    series_name = df.select("name").to_series().to_list()

    generated = []
    for i, cell in enumerate(series):
        pdb_id = parse_pdb_id(cell)
        if not pdb_id:
            continue
        try:
            out_path = make_figure_for_pdb(
                pdb_id=pdb_id,
                df_counts=df_counts,
                base_pnas_af3_lig=base_pnas_af3_lig,
                base_pnas_af3_nolig=base_pnas_af3_nolig,
                base_pdb_structures=base_pdb_structures,
                out_dir=out_dir,
                range=(series_min[i], series_max[i]),
                series_name=series_name[i],
            )
            generated.append(str(out_path))
            print(f"[OK] {pdb_id} -> {out_path}")
        except Exception as e:
            print(f"[WARN] Failed for {pdb_id}: {e}", file=sys.stderr)

    if not generated:
        print("[INFO] No figures generated.", file=sys.stderr)


if __name__ == "__main__":
    main()
