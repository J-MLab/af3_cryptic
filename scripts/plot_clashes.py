"""Script to plot clash histograms for different proteins based on clash summary CSV files.
Author: Maria L.
"""

from matplotlib import pyplot as plt
import polars as pl
from pathlib import Path
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import MaxNLocator

# --- Load data ---
csv_path_2 = Path("clash_summary_2.csv")
csv_path_25 = Path("clash_summary_25.csv")
correct_df = pl.read_csv(csv_path_2)
near_df = pl.read_csv(csv_path_25)

pnas_data = pl.read_csv("pnas_table.csv")
af_pdb_name_dict = {
    row["af_pdb"].split(".")[0]: row["name"] for row in pnas_data.to_dicts()
}

# --- Typography and styling ---
mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 17,
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "legend.fontsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

proteins = {
    "Bovine β-Lactoglobulin": "BLG",
    "KRAS": "K-RAS",
    "MAPK": "MAPK",
    "Pyruvate Dehydrogenase Kinase": "PDK",
    "Ribonuclease A": "RNase A",
    "β-Secretase": "BACE",
    "TEM β-lactamase": "TEM",
    "cAMP-dependent protein kinase": "PKA",
    "Glutamate receptor 2": "GluR2",
    "AmpC Beta-Lactamase": "AmpC",
    "Thrombin": "TT",
    "Adipocyte Lipid Droplet Binding Protein": "ALDBP",
    "Myosin II": "NM2",
    "Ricin": "RTA",
    "Androgen receptor": "AR",
    "Hsp90": "Hsp90",
}

# --- Collect unique PDBs ---
pdb_ids = near_df.select(pl.col("pdb_id").unique()).to_series().to_list()

# --- Create subplots ---
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True)
fig.supxlabel("Number of Residue Clashes", x=0.5, y=0.08, fontsize=17)

if len(pdb_ids) == 1:
    axes = [axes]
else:
    axes = np.ravel(axes)

# --- Plotting ---
bin_edges = np.arange(-0.5, 4.6, 1)  # Centered bins: -0.5→0.5, ..., 3.5→4.5

for ax, pdb in zip(axes, pdb_ids):
    df_correct = correct_df.filter(pl.col("pdb_id") == pdb)
    df_near = near_df.filter(pl.col("pdb_id") == pdb)

    correct_pose = df_correct["num_clashes"].to_numpy()
    near_correct_pose = df_near["num_clashes"].to_numpy()

    # Compute histograms manually for grouped bar plot
    counts_correct, _ = np.histogram(correct_pose, bins=bin_edges)
    counts_near, _ = np.histogram(near_correct_pose, bins=bin_edges)

    # Filter out bin 0
    bin_centers = np.arange(1, len(counts_correct))  # Start from 1 instead of 0
    counts_correct = counts_correct[1:]  # Remove first element
    counts_near = counts_near[1:]  # Remove first element

    width = 0.35
    ax.bar(
        bin_centers - width / 2,
        counts_correct,
        width=width,
        label="< 2.0 Å",
        color="#76acd8",
        edgecolor="black",
    )
    ax.bar(
        bin_centers + width / 2,
        counts_near,
        width=width,
        label="< 2.5 Å",
        color="#dfd1d1",
        edgecolor="black",
    )

    ax.set_title(f"{proteins[af_pdb_name_dict[pdb.upper()]]}", pad=6)
    ax.set_ylabel("Structure count", labelpad=6)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([str(i) for i in bin_centers], rotation=0)
    ax.tick_params(axis="x", which="major", labelrotation=0)

    ax.set_xlim(0.5, max(bin_centers) + 0.5)
    ax.set_ylim(0, 32)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.minorticks_off()
    ax.tick_params(direction="in", length=3)

# --- Legend outside the grid ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    ncol=1,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.93, 0.90),  # (x, y) relative to figure
)

fig.tight_layout()
fig.subplots_adjust(top=0.92, bottom=0.20)
fig.align_labels()

# --- Save ---
Path("plots_out").mkdir(parents=True, exist_ok=True)
plt.savefig("plots_out/clash_histogram.png", bbox_inches="tight")
plt.close()
