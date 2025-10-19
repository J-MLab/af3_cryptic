#!/usr/bin/env python3
"""Script to plot ligand pose RMSD vs ligand pLDDT for multiple proteins
Author: Maria L.
"""

import os
import re
from pathlib import Path
import argparse
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns  # gives us access to large categorical palettes
import numpy as np

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


def extract_label_from_filename(path):
    base = os.path.basename(path)
    m = re.match(r"^([0-9A-Za-z]+)_lig_RMSDs.*\.csv$", base)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]


def main():
    pose_path = Path("pose_rmsds")
    df_info = pl.read_csv("pnas_table.csv")
    df_info = df_info.with_columns(pl.col("af_pdb").str.to_lowercase())
    af_pdbs = df_info["af_pdb"].to_list()

    all_files = sorted(pose_path.glob("*_lig_RMSDs_with_pLDDT.csv"))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(all_files) + 1)))

    # Build a distinct color palette (one color per protein)
    n_proteins = len(proteins)
    palette = sns.color_palette("tab20", n_colors=n_proteins)
    protein_to_color = dict(zip(proteins.values(), palette))

    # Collect ALL data points for global regression
    all_x, all_y = [], []

    for afpdb in af_pdbs:
        print(afpdb)
        for f in all_files:
            if afpdb[:-2] in f.name:
                label = extract_label_from_filename(f)

                match = df_info.filter(
                    pl.col("af_pdb").str.slice(0, 4).str.to_lowercase() == label.lower()
                )
                label = proteins[match["name"].to_list()[0]]

                df_rmsd = pl.read_csv(f)
                if "lig_RMSD" not in df_rmsd.columns:
                    continue

                yvals = df_rmsd["lig_RMSD"].cast(pl.Float64).drop_nulls().to_list()
                xvals = df_rmsd["pLDDT"].cast(pl.Float64).drop_nulls().to_list()

                # Add to global lists
                all_x.extend(xvals)
                all_y.extend(yvals)

                # Per-protein scatter
                ax.scatter(
                    xvals,
                    yvals,
                    edgecolors="black",
                    linewidths=1,
                    s=200,
                    label=label,
                    color=protein_to_color[label],
                )

    # --- Global regression across ALL points ---
    if all_x and all_y:
        m, b = np.polyfit(all_x, all_y, 1)  # linear fit
        x_line = np.linspace(min(all_x), max(all_x), 100)
        y_line = m * x_line + b

        # R² calculation
        y_pred = m * np.array(all_x) + b
        ss_res = np.sum((np.array(all_y) - y_pred) ** 2)
        ss_tot = np.sum((np.array(all_y) - np.mean(all_y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        ax.plot(
            x_line,
            y_line,
            color="red",
            linewidth=2,
            linestyle="--",
            label=f"Linear (R² = {r2:.2f})",
        )

    # Horizontal threshold line
    ax.axhline(y=2, linestyle="--", linewidth=2, color="black", alpha=0.6)

    # Axis labels and ticks
    ax.set_xlabel("Ligand pLDDT", fontsize=20)
    ax.set_ylabel("Pose RMSD (Å)", fontsize=20)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_yscale("log")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_xticks(range(45, 101, 5))
    ax.tick_params(axis="x", labelsize=16)
    ax.set_xlim(45, 100)
    custom_ticks = [1, 2, 2.5, 5, 10, 20, 30]
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels([str(t) for t in custom_ticks])

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        fontsize=12,
        title="Protein",
        title_fontsize=14,
        ncol=2,
    )

    fig.tight_layout()

    out_dir = os.path.dirname("plots_out/pose_rmsds.png")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig("plots_out/rmsd_vs_plddt.png", dpi=500)


if __name__ == "__main__":
    main()
