#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import matplotlib.cm as cm

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

proteins = {
    'Rho guanine nucleotide exchange factor 2': 'ARHGEF2',
    'Formate-tetrahydrofolate ligase':'FTHFS',
    'Dihydrofolate reductase':'DHFR',
    'D-alanine--D-alanine ligase A':'DDL',
    'Probable cytosol aminopeptidase':'pepA',
    '2,3-dihydroxybenzoate-2,3-dehydrogenase':'DhbA',
    'Unc119':'Unc119',
    'UDP-glycosyltransferase':'UGT',
    'Protease 3C':'3Cpro',
    'Polyunsaturated fatty acid lipoxygenase ALOX12':'12-LOX'
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


def calculate_rmsd_difference(closed_rmsd: np.ndarray, open_rmsd: np.ndarray) -> np.ndarray:
    """
    Calculate RMSD_open - RMSD_closed for each structure.
    Returns array of differences. Empty if either input is empty.
    Filters out NA/NaN values.
    """
    if closed_rmsd.size == 0 or open_rmsd.size == 0:
        return np.array([])
    n = min(closed_rmsd.size, open_rmsd.size)
    
    # Calculate differences
    diff = open_rmsd[:n] - closed_rmsd[:n]
    
    # Remove NA/NaN values
    diff = diff[~np.isnan(diff)]
    
    return diff

def grouped_boxplot(ax, data_dict, uniform_color, title, y_range):
    """
    Create a cleaner grouped boxplot for multiple proteins and conditions.
    All boxes use the same color.
    
    Args:
        ax: matplotlib axis
        data_dict: dict of {protein_label: np.array of differences}
        uniform_color: single color to use for all boxes
        title: plot title
        y_range: tuple of (min, max) for y-axis
    """
    # Filter out empty datasets
    valid_data = []
    valid_labels = []
    
    for label in data_dict.keys():
        data = data_dict[label]
        if data.size > 0:
            valid_data.append(data)
            valid_labels.append(label)
    
    if not valid_data:
        annotate_missing(ax, "No data available")
        return
    
    # Create boxplot with cleaner styling
    bp = ax.boxplot(
        valid_data,
        tick_labels=valid_labels,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        # Make outliers smaller and more subtle
        flierprops=dict(
            marker='o', 
            markersize=2.5, 
            alpha=0.4,
            markeredgewidth=0
        ),
        # Cleaner box styling
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        medianprops=dict(linewidth=1.5, color='darkred')
    )
    
    # Color all boxes with the same uniform color
    for patch in bp['boxes']:
        patch.set_facecolor(uniform_color)
        patch.set_alpha(0.7)
    
    # Clean styling - remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Simple, clear y-label (removed the sign convention from here)
    ax.set_ylabel("RMSD Difference (Å)", fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Subtle grid only on y-axis
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)  # Grid behind data
    
    # Reference line at y=0 - make it more prominent
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.7, zorder=1)
    
    if y_range:
        ax.set_ylim(y_range)
    
    # Rotate x-axis labels for readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add legend to explain the sign convention
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='', color='w', label='− open', markersize=0),
        Line2D([0], [0], marker='', color='w', label='+ closed', markersize=0)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
              frameon=True, fancybox=False, edgecolor='black', handlelength=0, handletextpad=0)
    
def bar_states_aggregated(ax, all_states_data, title):
    """
    Create aggregated bar chart for multiple proteins in a group.
    all_states_data: list of tuples (states, values) for each protein
    """
    colors = ["#76acd8", "#dfd1d1", "#746d72"]  # blue, gray, purple
    
    # Aggregate counts across all proteins
    if not all_states_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
        return
    
    # Sum up all values
    states = ["closed", "open", "other"]
    totals = [0.0, 0.0, 0.0]
    
    for st_list, val_list in all_states_data:
        for i, val in enumerate(val_list):
            totals[i] += val
    
    bars = ax.bar(
        states,
        totals,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_ylabel("Count", fontsize=16)
    ax.set_xlabel("Conformation", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    max_val = max(totals) if totals else 1
    ax.set_ylim(0, max_val * 1.15)
    ax.set_title(f'{title}', fontsize=22)

    # add value labels on top of bars
    for bar, val in zip(bars, totals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(totals) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )


def make_figure_for_group(
    group_name: str,
    group_data: list,
    df_counts: pl.DataFrame,
    base_pnas_af3_lig: Path,
    base_pnas_af3_nolig: Path,
    base_pdb_structures: Path,
    out_dir: Path,
):
    """
    Build 1x4 grid for given group and save to out_dir/group_name.png
    Row 1: 4 boxplots for each condition
    
    group_data: list of dicts with keys: pdb_id, name, min, max
    """
    # Create figure with better spacing
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=200)
    plt.subplots_adjust(wspace=0.25, hspace=0.30)

    # Choose uniform color: dull baby blue or beige
    # Baby blue option: '#9DB4C0' or '#A8C5D6' or '#B0C4DE' (lighter)
    # Beige option: '#D4C5B9' or '#E0D5C7' or '#C9B8A0'
    
    uniform_color = '#A8C5D6'  # Dull baby blue
    # uniform_color = '#D4C5B9'  # Beige - uncomment to use this instead

    # Collect data for all proteins in the group
    af3_lig_data = {}
    pdb_bound_data = {}
    af3_nolig_data = {}
    pdb_unbound_data = {}
    
    for item in group_data:
        pdb_id = item['pdb_id'].lower()
        protein_label = proteins.get(item['name'], item['name'])
        
        # (1) AF3 with ligand
        lig_dir = base_pnas_af3_lig / pdb_id.lower()
        af3_lig_closed = safe_read_loop_rmsd(lig_dir / "closed_rmsd.csv")
        af3_lig_open = safe_read_loop_rmsd(lig_dir / "open_rmsd.csv")
        af3_lig_diff = calculate_rmsd_difference(af3_lig_closed, af3_lig_open)
        if af3_lig_diff.size > 0:
            af3_lig_data[protein_label] = af3_lig_diff
        
        # (2) PDB bound
        bound_dir = base_pdb_structures / f"{pdb_id}_pdbs" / "bound"
        pdb_bound_closed = safe_read_loop_rmsd(bound_dir / "closed_rmsd.csv")
        pdb_bound_open = safe_read_loop_rmsd(bound_dir / "open_rmsd.csv")
        pdb_bound_diff = calculate_rmsd_difference(pdb_bound_closed, pdb_bound_open)
        if pdb_bound_diff.size > 0:
            pdb_bound_data[protein_label] = pdb_bound_diff
        
        # (3) AF3 without ligand
        nolig_dir = base_pnas_af3_nolig / pdb_id.lower()
        af3_nolig_closed = safe_read_loop_rmsd(nolig_dir / "closed_rmsd.csv")
        af3_nolig_open = safe_read_loop_rmsd(nolig_dir / "open_rmsd.csv")
        af3_nolig_diff = calculate_rmsd_difference(af3_nolig_closed, af3_nolig_open)
        if af3_nolig_diff.size > 0:
            af3_nolig_data[protein_label] = af3_nolig_diff
        
        # (4) PDB unbound
        unbound_dir = base_pdb_structures / f"{pdb_id}_pdbs" / "unbound"
        pdb_unbound_closed = safe_read_loop_rmsd(unbound_dir / "closed_rmsd.csv")
        pdb_unbound_open = safe_read_loop_rmsd(unbound_dir / "open_rmsd.csv")
        pdb_unbound_diff = calculate_rmsd_difference(pdb_unbound_closed, pdb_unbound_open)
        if pdb_unbound_diff.size > 0:
            pdb_unbound_data[protein_label] = pdb_unbound_diff
    
    # Calculate global y-axis range from actual data across all conditions
    all_data_values = []
    
    for data_dict in [af3_lig_data, pdb_bound_data, af3_nolig_data, pdb_unbound_data]:
        for protein_label, data in data_dict.items():
            if data.size > 0:
                all_data_values.extend(data.tolist())
    
    y_range = None
    if all_data_values:
        min_val = np.min(all_data_values)
        max_val = np.max(all_data_values)
        # Add some padding (10% on each side)
        padding = (max_val - min_val) * 0.1
        y_range = (min_val - padding, max_val + padding)
    
    # Create boxplots - one for each condition
    grouped_boxplot(
        axes[0],
        data_dict=af3_lig_data,
        uniform_color=uniform_color,
        title="AF3 w/ Ligand",
        y_range=y_range
    )
    
    grouped_boxplot(
        axes[1],
        data_dict=pdb_bound_data,
        uniform_color=uniform_color,
        title="PDB Bound",
        y_range=y_range
    )
    
    grouped_boxplot(
        axes[2],
        data_dict=af3_nolig_data,
        uniform_color=uniform_color,
        title="AF3 w/o Ligand",
        y_range=y_range
    )
    
    grouped_boxplot(
        axes[3],
        data_dict=pdb_unbound_data,
        uniform_color=uniform_color,
        title="PDB Unbound",
        y_range=y_range
    )

    # Only show y-label on leftmost plot
    for ax in axes[1:]:
        ax.set_ylabel('')


    # plt.suptitle(f"Group {group_name}", fontsize=16, fontweight='bold', y=0.98)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{group_name}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------- Main ----------


def main():
    df_csv = Path("unbiased_table_cut.csv")
    counts_csv = Path("total_state_counts_no_cutoff.csv")
    base_pnas_af3_lig = Path("af3_lig_unbiased")
    base_pnas_af3_nolig = Path("af3_nolig_unbiased")
    base_pdb_structures = Path("pdb_structures_no_cutoff")
    out_dir = Path("plots_out")

    # Load inputs
    try:
        df = pl.read_csv(df_csv, infer_schema_length=10000)
    except Exception as e:
        print(f"[ERROR] Failed to read df_csv: {df_csv} -> {e}", file=sys.stderr)
        sys.exit(1)

    # Check for required columns
    required_cols = ["af_pdb", "group", "name"]
    for col in required_cols:
        if col not in df.columns:
            # try case-insensitive match
            candidates = [c for c in df.columns if c.lower() == col.lower()]
            if not candidates:
                print(f"[ERROR] Input dataframe must have a '{col}' column.", file=sys.stderr)
                sys.exit(1)
            df = df.rename({candidates[0]: col})
    
    # Check for optional min/max columns
    if "min" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("min"))
    if "max" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("max"))

    try:
        df_counts = read_counts_table(counts_csv)
    except Exception as e:
        print(
            f"[ERROR] Failed to read counts_csv: {counts_csv} -> {e}", file=sys.stderr
        )
        sys.exit(1)

    # Group by 'group' column - now expecting 1a, 1b, 1c, 2, 3
    groups = df.select("group").unique().to_series().to_list()
    group_names = ['1a', '1b', '1c', '2', '3']

    generated = []
    for group in groups:
        if not group or str(group).lower() == 'null':
            continue
            
        # Get all proteins in this group
        group_df = df.filter(pl.col("group") == group)
        
        group_data = []
        for row in group_df.iter_rows(named=True):
            pdb_id = parse_pdb_id(row['af_pdb'])
            if not pdb_id:
                continue
            group_data.append({
                'pdb_id': pdb_id,
                'name': row['name'],
                'min': row.get('min'),
                'max': row.get('max'),
            })
        
        if not group_data:
            print(f"[WARN] No valid PDB IDs in group '{group}'", file=sys.stderr)
            continue

        try:
            out_path = make_figure_for_group(
                group_name=group_names[int(group)-1],
                group_data=group_data,
                df_counts=df_counts,
                base_pnas_af3_lig=base_pnas_af3_lig,
                base_pnas_af3_nolig=base_pnas_af3_nolig,
                base_pdb_structures=base_pdb_structures,
                out_dir=out_dir,
            )
            generated.append(str(out_path))
            print(f"[OK] Group '{group}' -> {out_path}")
        except Exception as e:
            print(f"[WARN] Failed for group '{group}': {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    if not generated:
        print("[INFO] No figures generated.", file=sys.stderr)
    else:
        print(f"[INFO] Generated {len(generated)} figure(s).")


if __name__ == "__main__":
    main()