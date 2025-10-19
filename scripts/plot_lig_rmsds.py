#!/usr/bin/env python3
"""Script to plot ligand RMSDs for different proteins from AF3 cryptic site modeling.
Author: Maria L.
"""

import os
import re
from pathlib import Path
import argparse
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

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

    marker_cycle = {
        "Overlaps with active site": "o",
        "Fully separate cryptic": "s",
        "Adjacent to active": "^",
    }
    type_to_marker = {}
    type_seen_global = set()

    y_labels = []
    y_positions = {}
    current_y = 1

    minim = 100
    n_proteins = len(proteins)
    palette = sns.color_palette("tab20", n_colors=n_proteins)
    protein_to_color = dict(zip(proteins.values(), palette))

    for afpdb in reversed(af_pdbs):
        print(afpdb)
        for f in all_files:
            if afpdb[:-2] in f.name:
                label = extract_label_from_filename(f)
                # Try to get a more descriptive label from df_info['name'] where 'af_pdb' == label
                match = df_info.filter(
                    pl.col("af_pdb")
                    .str.slice(0, 4)  # Extract the 4-character PDB ID
                    .str.to_lowercase()  # Make it lowercase
                    == label.lower()  # Compare to input (lowercased)
                )
                label = proteins[match["name"].to_list()[0]]
                y_labels.append(proteins[match["name"].to_list()[0]])
                y_positions[label] = current_y

                df_rmsd = pl.read_csv(f)
                if "lig_RMSD" not in df_rmsd.columns:
                    continue
                xvals = df_rmsd["lig_RMSD"].cast(pl.Float64)
                xvals = xvals.drop_nulls()
                confidences = df_rmsd["pLDDT"].cast(pl.Float64)
                confidences = confidences.drop_nulls()
                if min(confidences) < minim:
                    minim = min(confidences)
                type = match["type"].to_list()[0]

                sc = ax.scatter(
                    xvals,
                    [current_y] * len(xvals),
                    marker=marker_cycle[type],
                    edgecolors="black",
                    linewidths=1,
                    s=200,
                    label=type,
                    color=protein_to_color[label],
                )

                current_y += 1

    # Vertical lines
    ax.axvline(x=2, linestyle="--", linewidth=2, color="black", alpha=0.6)

    ax.set_yticks(range(1, len(y_labels) + 1))
    ax.set_yticklabels(y_labels, fontsize=20)
    ax.set_xticklabels(range(-5, 45, 5), fontsize=16)
    ax.set_xlabel("Pose RMSD (Å)", fontsize=20)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower right",
        title="Cryptic Site Type",
        fontsize=14,
        title_fontsize=16,
    )
    fig.tight_layout()

    out_dir = os.path.dirname("plots_out/pose_rmsds.png")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig("plots_out/pose_rmsds.png", dpi=500)


if __name__ == "__main__":
    main()
