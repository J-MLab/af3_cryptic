#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import polars as pl
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import rcParams

# ---------- Publication-quality global defaults ----------
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Palatino"],
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "lines.linewidth": 1.0,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#444444",
    "pdf.fonttype": 42,   # embeds fonts for vector export
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})

# ---------- Constants ----------

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
    'Formate-tetrahydrofolate ligase': 'FTHFS',
    'Dihydrofolate reductase': 'DHFR',
    'D-alanine--D-alanine ligase A': 'DDL',
    'Probable cytosol aminopeptidase': 'pepA',
    '2,3-dihydroxybenzoate-2,3-dehydrogenase': 'DhbA',
    'Unc119': 'Unc119',
    'UDP-glycosyltransferase': 'UGT',
    'Protease 3C': '3Cpro',
    'Polyunsaturated fatty acid lipoxygenase ALOX12': '12-LOX',
}

GROUP_COLORS = {
    "1b": '#1A4971',  # Navy Blue
    "1c": '#2E86C1',  # Normal Blue
    "2":  '#D6E8F0',  # Light Blue
    "3": '#D6E8F0'    # Light Blue
}

# Single color for solo-group plots (1a, 3)
SOLO_COLOR = "#76acd8"

# ---------- Helpers ----------

def parse_pdb_id(cell: str) -> str:
    if not cell:
        return None
    return str(cell).split(".")[0].upper()


def read_counts_table(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Counts CSV not found: {path}")
    df = pl.read_csv(path, infer_schema_length=10000)
    if not {"State", "Count"}.issubset(df.columns):
        raise ValueError(f"CSV must contain 'State' and 'Count' columns, found {df.columns}")
    df = df.select([
        pl.col("State").cast(pl.Utf8).alias("key"),
        pl.col("Count").cast(pl.Float64).alias("value"),
    ])
    return df


def safe_read_loop_rmsd(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        return np.array([])
    try:
        df = pl.read_csv(csv_path, infer_schema_length=10000)
        if "loop_rmsd" in df.columns:
            vals = df.select(pl.col("loop_rmsd")).to_series().to_numpy()
            return np.array(vals, dtype=float)
        lower_cols = {c.lower(): c for c in df.columns}
        if "loop_rmsd" in lower_cols:
            vals = df.select(pl.col(lower_cols["loop_rmsd"])).to_series().to_numpy()
            return np.array(vals, dtype=float)
        return np.array([])
    except Exception:
        return np.array([])


def annotate_missing(ax, msg: str = "No data available"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9,
            color="#888888", style="italic", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def calculate_rmsd_difference(closed_rmsd: np.ndarray, open_rmsd: np.ndarray) -> np.ndarray:
    if closed_rmsd.size == 0 or open_rmsd.size == 0:
        return np.array([])
    n = min(closed_rmsd.size, open_rmsd.size)
    diff = open_rmsd[:n] - closed_rmsd[:n]
    return diff[~np.isnan(diff)]


def load_rmsd_diffs_for_item(item, base_pnas_af3_lig, base_pnas_af3_nolig, base_pdb_structures):
    """
    For a single protein dict, load all four condition diffs.
    Returns dict keyed by condition name -> np.ndarray.
    """
    pdb_id = item['pdb_id'].lower()
    out = {}

    lig_dir = base_pnas_af3_lig / pdb_id
    c = safe_read_loop_rmsd(lig_dir / "closed_rmsd.csv")
    o = safe_read_loop_rmsd(lig_dir / "open_rmsd.csv")
    out['af3_lig'] = calculate_rmsd_difference(c, o)

    bound_dir = base_pdb_structures / f"{pdb_id}_pdbs" / "bound"
    c = safe_read_loop_rmsd(bound_dir / "closed_rmsd.csv")
    o = safe_read_loop_rmsd(bound_dir / "open_rmsd.csv")
    out['pdb_bound'] = calculate_rmsd_difference(c, o)

    nolig_dir = base_pnas_af3_nolig / pdb_id
    c = safe_read_loop_rmsd(nolig_dir / "closed_rmsd.csv")
    o = safe_read_loop_rmsd(nolig_dir / "open_rmsd.csv")
    out['af3_nolig'] = calculate_rmsd_difference(c, o)

    unbound_dir = base_pdb_structures / f"{pdb_id}_pdbs" / "unbound"
    c = safe_read_loop_rmsd(unbound_dir / "closed_rmsd.csv")
    o = safe_read_loop_rmsd(unbound_dir / "open_rmsd.csv")
    out['pdb_unbound'] = calculate_rmsd_difference(c, o)

    return out


# ---------- Boxplot drawing ----------

def _style_ax(ax, title, show_ylabel=True, y_range=None):
    """Apply consistent publication styling to an axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='both', direction='out', length=3, width=0.8)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4, linewidth=0.6, color='#888888')
    ax.set_axisbelow(True)
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.8, alpha=0.8, zorder=2)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8, loc='center')
    if show_ylabel:
        ax.set_ylabel("RMSD$_{open}$ − RMSD$_{closed}$ (Å)", fontsize=10)
    if y_range:
        ax.set_ylim(y_range)
    # Sign-convention legend on every axis
    sign_handles = [
        mpatches.Patch(color='none', label='− open'),
        mpatches.Patch(color='none', label='+ closed'),
    ]
    ax.legend(
        handles=sign_handles,
        loc='upper right',
        fontsize=8,
        frameon=True,
        framealpha=0.85,
        edgecolor='#888888',
        handlelength=0,
        handletextpad=0,
        borderpad=0.5,
    )


def grouped_boxplot_single(ax, data_dict, color, title, y_range, show_ylabel=True):
    """
    Boxplot for a single group (all boxes same color).
    data_dict: {protein_label: np.ndarray}
    """
    valid_data = [(lbl, d) for lbl, d in data_dict.items() if d.size > 0]
    if not valid_data:
        annotate_missing(ax)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        return

    labels, arrays = zip(*valid_data)

    bp = ax.boxplot(
        arrays,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        flierprops=dict(marker='o', markersize=2.0, alpha=0.35,
                        markeredgewidth=0, markerfacecolor=color),
        boxprops=dict(linewidth=0.9),
        whiskerprops=dict(linewidth=0.9, linestyle='--'),
        capprops=dict(linewidth=0.9),
        medianprops=dict(linewidth=1.5, color=color),
    )
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    _style_ax(ax, title, show_ylabel=show_ylabel, y_range=y_range)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=40, ha='right', fontsize=9)


def grouped_boxplot_multi(ax, data_by_group, color_map, title, y_range, show_ylabel=True):
    """
    Boxplot where proteins come from multiple sub-groups, each with its own color.
    data_by_group: list of (group_name, protein_label, np.ndarray)
    color_map: {group_name: color}
    Returns list of (group_name, color) for building the legend.
    """
    valid = [(g, lbl, d) for g, lbl, d in data_by_group if d.size > 0]
    if not valid:
        annotate_missing(ax)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        return []

    groups_seen, labels, arrays, colors_per_box = [], [], [], []
    for g, lbl, d in valid:
        groups_seen.append(g)
        labels.append(lbl)
        arrays.append(d)
        colors_per_box.append(color_map[g])

    bp = ax.boxplot(
        arrays,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        flierprops=dict(marker='o', markersize=2.0, alpha=0.35, markeredgewidth=0),
        boxprops=dict(linewidth=0.9),
        whiskerprops=dict(linewidth=0.9, linestyle='--'),
        capprops=dict(linewidth=0.9),
        medianprops=dict(linewidth=1.5, color='black'),  # overridden per-box below
    )
    for patch, c in zip(bp['boxes'], colors_per_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)

    # Set each median line to its box color (fully opaque so it stays visible)
    for median_line, c in zip(bp['medians'], colors_per_box):
        median_line.set_color(c)
        median_line.set_alpha(1.0)

    # Color fliers to match their box
    if 'fliers' in bp:
        for flier, c in zip(bp['fliers'], colors_per_box):
            flier.set_markerfacecolor(c)

    _style_ax(ax, title, show_ylabel=show_ylabel, y_range=y_range)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=40, ha='right', fontsize=9)

    return list(dict.fromkeys(zip(groups_seen, colors_per_box)))  # ordered unique


# ---------- Figure builders ----------

CONDITION_TITLES = [
    ("af3_lig",    "AF3 w/ Ligand"),
    ("pdb_bound",  "PDB Bound"),
    ("af3_nolig",  "AF3 w/o Ligand"),
    ("pdb_unbound","PDB Unbound"),
]


def make_figure_single_group(group_name, group_data, paths, out_dir):
    """
    Single group (1a or 3): all boxes in SOLO_COLOR, no group legend needed.
    """
    base_af3_lig, base_af3_nolig, base_pdb = paths

    # Collect per-condition data dicts
    condition_data = {cond: {} for cond, _ in CONDITION_TITLES}
    for item in group_data:
        label = proteins.get(item['name'], item['name'])
        diffs = load_rmsd_diffs_for_item(item, base_af3_lig, base_af3_nolig, base_pdb)
        for cond, _ in CONDITION_TITLES:
            if diffs[cond].size > 0:
                condition_data[cond][label] = diffs[cond]

    y_range = _global_y_range(condition_data)
    # Extend top by 1 to give the sign-convention legend room
    if y_range:
        y_range = (y_range[0], y_range[1] + 1)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), dpi=300)
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.97, top=0.88, bottom=0.20)

    for i, (cond, title) in enumerate(CONDITION_TITLES):
        grouped_boxplot_single(
            axes[i], condition_data[cond], SOLO_COLOR,
            title=title, y_range=y_range, show_ylabel=(i == 0)
        )

    # fig.suptitle(f"Group {group_name}", fontsize=14, fontweight='bold', y=0.97)
    _save(fig, out_dir, group_name)


def make_figure_multi_group(combined_name, subgroups_data, color_map, paths, out_dir):
    """
    Multiple sub-groups (1b+1c+2): boxes colored by sub-group with a legend.
    subgroups_data: list of (sub_group_name, [item dicts])
    """
    base_af3_lig, base_af3_nolig, base_pdb = paths

    # Collect per-condition flat lists of (sub_group, label, diff)
    condition_data = {cond: [] for cond, _ in CONDITION_TITLES}
    for sub_group, items in subgroups_data:
        for item in items:
            label = proteins.get(item['name'], item['name'])
            diffs = load_rmsd_diffs_for_item(item, base_af3_lig, base_af3_nolig, base_pdb)
            for cond, _ in CONDITION_TITLES:
                condition_data[cond].append((sub_group, label, diffs[cond]))

    # Compute global y range from all valid data
    all_vals = []
    for cond_list in condition_data.values():
        for _, _, d in cond_list:
            if d.size > 0:
                all_vals.extend(d.tolist())
    y_range = _y_range_from_vals(all_vals)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), dpi=300)
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.97, top=0.84, bottom=0.22)

    legend_entries = None
    for i, (cond, title) in enumerate(CONDITION_TITLES):
        entries = grouped_boxplot_multi(
            axes[i], condition_data[cond], color_map,
            title=title, y_range=y_range, show_ylabel=(i == 0)
        )
        if entries and legend_entries is None:
            legend_entries = entries

    # Add group color legend below the subplots
    if legend_entries:
        handles = [
            mpatches.Patch(facecolor=c, edgecolor='#444444', linewidth=0.8,
                           alpha=0.72, label=f"Group {g}")
            for g, c in legend_entries
        ]
        # fig.legend(
        #     handles=handles,
        #     loc='lower center',
        #     ncol=len(handles),
        #     bbox_to_anchor=(0.52, -0.01),
        #     frameon=True,
        #     framealpha=0.92,
        #     edgecolor='#888888',
        #     fontsize=9,
        #     title="Group",
        #     title_fontsize=9,
        # )

    # fig.suptitle(f"Groups 1b, 1c and 2", fontsize=14, fontweight='bold', y=0.97)
    _save(fig, out_dir, combined_name)


# ---------- Utilities ----------

def _global_y_range(condition_data):
    all_vals = []
    for d in condition_data.values():
        for arr in d.values():
            if arr.size > 0:
                all_vals.extend(arr.tolist())
    return _y_range_from_vals(all_vals)


def _y_range_from_vals(vals):
    if not vals:
        return None
    lo, hi = np.min(vals), np.max(vals)
    pad = max((hi - lo) * 0.12, 0.05)
    return (lo - pad, hi + pad)


def _save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_dir / name}.png/.pdf")


# ---------- Main ----------

def main():
    df_csv              = Path("pnas_table_mod.csv")
    counts_csv          = Path("total_state_counts_mod.csv")
    base_pnas_af3_lig   = Path("pnas_af3_lig")
    base_pnas_af3_nolig = Path("pnas_af3_nolig")
    base_pdb_structures = Path("pdb_structures_final")
    out_dir             = Path("plots_out")

    df_csv              = Path("unbiased_table_cut.csv")
    counts_csv          = Path("total_state_counts_no_cutoff.csv")
    base_pnas_af3_lig   = Path("af3_lig_unbiased")
    base_pnas_af3_nolig = Path("af3_nolig_unbiased")
    base_pdb_structures = Path("pdb_structures_no_cutoff")
    out_dir             = Path("plots_out")

    paths = (base_pnas_af3_lig, base_pnas_af3_nolig, base_pdb_structures)

    try:
        df = pl.read_csv(df_csv, infer_schema_length=10000)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = ["af_pdb", "group", "name"]
    for col in required_cols:
        if col not in df.columns:
            candidates = [c for c in df.columns if c.lower() == col.lower()]
            if not candidates:
                print(f"[ERROR] Missing column '{col}'", file=sys.stderr)
                sys.exit(1)
            df = df.rename({candidates[0]: col})

    for opt in ("min", "max"):
        if opt not in df.columns:
            df = df.with_columns(pl.lit(None).alias(opt))

    # Ensure group column is string so string comparisons like '1a' work
    df = df.with_columns(pl.col("group").cast(pl.Utf8))

    def rows_for_group(g):
        rows = []
        for row in df.filter(pl.col("group") == g).iter_rows(named=True):
            pdb_id = parse_pdb_id(row['af_pdb'])
            if pdb_id:
                rows.append({'pdb_id': pdb_id, 'name': row['name'],
                             'min': row.get('min'), 'max': row.get('max')})
        return rows

    # --- Plot 1: Group 1a (solo) ---
    data_1a = rows_for_group('1a')
    if data_1a:
        make_figure_single_group('1a', data_1a, paths, out_dir)
    else:
        print("[WARN] No data for group 1a")

    # --- Plot 2: Groups 1b + 1c + 2 (combined, color-coded) ---
    combined_subgroups = []
    # for g in ('1b', '1c', '2'):
    for g in ('1b', '1c', '3'):
        d = rows_for_group(g)
        if d:
            combined_subgroups.append((g, d))
        else:
            print(f"[WARN] No data for group {g}")

    if combined_subgroups:
        make_figure_multi_group(
            combined_name="1b_1c_2",
            subgroups_data=combined_subgroups,
            color_map=GROUP_COLORS,
            paths=paths,
            out_dir=out_dir,
        )

    # --- Plot 3: Group 3 (solo) ---
    data_3 = rows_for_group('3')
    if data_3:
        make_figure_single_group('3', data_3, paths, out_dir)
    else:
        print("[WARN] No data for group 3")


if __name__ == "__main__":
    main()