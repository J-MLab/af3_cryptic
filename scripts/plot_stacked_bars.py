#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacked bar plots of conformational state counts per protein, per condition.

Figure layout
-------------
- One ROW per group (1a, 1b, 1c, 2, 3, ...).
- One COLUMN per condition (AF3 w/ Ligand, PDB Bound, AF3 w/o Ligand, PDB Unbound).
- Each BAR is one protein, stacked by state (closed / open / other). For
  ligand-bearing conditions, "open" is further split into open_correct_pose /
  open_incorrect_pose when a pose-RMSD file is available.

State assignment (per model)
----------------------------
A model is classified by reading its own row from `closed_rmsd.csv` and
`open_rmsd.csv` in the relevant directory:

    if loop_rmsd_to_closed <= threshold: closed
    elif loop_rmsd_to_open  <= threshold: open
    else:                                  other

Threshold = find_rmsd(pdb) / 2, matching the original counting script. For each
protein the threshold equals half the open PDB's loop_rmsd vs. the closed
reference (read from `pdb_structures_no_cutoff/<pdb>_pdbs/bound/closed_rmsd.csv`,
or `unbound/closed_rmsd.csv` for the 3IXJ special case).

Pose RMSD (open-state split)
----------------------------
For the AF3-w/-Ligand condition only, each "open" model is further split:

    open_correct   if pose_rmsd <= POSE_RMSD_CUTOFF
    open_incorrect if pose_rmsd >  POSE_RMSD_CUTOFF

If the pose-RMSD file is missing for a protein, its open models stay in a
single plain "open" bucket.

Segment labels
--------------
Each non-zero stacked segment is annotated with its RAW count, but ONLY when
that count is greater than `label_min_count` (default 50). Segments with 50 or
fewer models are left unlabeled. There is no longer a "promote to total label"
fallback.

Modularity
----------
Everything you'd want to toggle lives in `PlotConfig` at the bottom:
- Drop PDB columns  -> remove entries from `conditions`.
- Disable open-pose split  -> `split_open=False`.
- Change colors, state order, pose cutoff, normalization, etc. -- one source
  of truth.

Polars only. No pandas.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable

import polars as pl
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams


# ===========================================================================
# Global style — matches the boxplot script
# ===========================================================================

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
    "lines.linewidth": 1.0,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#444444",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


# ===========================================================================
# Protein label map (short names for x-axis ticks)
# ===========================================================================

PROTEIN_LABELS = {
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
    "Rho guanine nucleotide exchange factor 2": "ARHGEF2",
    "Formate-tetrahydrofolate ligase": "FTHFS",
    "Dihydrofolate reductase": "DHFR",
    "D-alanine--D-alanine ligase A": "DDL",
    "Probable cytosol aminopeptidase": "pepA",
    "2,3-dihydroxybenzoate-2,3-dehydrogenase": "DhbA",
    "Unc119": "Unc119",
    "UDP-glycosyltransferase": "UGT",
    "Protease 3C": "3Cpro",
    "Polyunsaturated fatty acid lipoxygenase ALOX12": "12-LOX",
}


# ===========================================================================
# Color palette
# ===========================================================================

COLOR_OPEN           = "#76acd8"  # open
COLOR_CLOSED         = "#dfd1d1"  # closed
COLOR_OTHER          = "#746d72"  # other

# Shades of the open color, used when split_open=True.
# Edit these freely if you prefer different shades.
COLOR_OPEN_CORRECT   = "#76acd8"  # same as plain open
COLOR_OPEN_INCORRECT = "#b8d3e9"  # lighter version of open


# ===========================================================================
# Configuration dataclasses
# ===========================================================================

@dataclass
class StateSpec:
    """One stacked segment in a bar."""
    name: str        # internal canonical key
    label: str       # legend label
    color: str       # hex color


@dataclass
class ConditionSpec:
    """One column in the figure (one condition)."""
    key: str
    title: str
    folder_resolver: Callable        # (pdb_lower, pdb_upper, paths) -> Path
    has_pose: bool = False           # ligand pose data exists for this condition?

    # Per-condition segment-label rule:
    #   "count"         -> label a segment only if its raw count > label_min_count
    #                      (label_min_count comes from PlotConfig)
    #   "ylim_fraction" -> label a segment only if its plotted height exceeds
    #                      label_ylim_fraction of the axis y-range
    label_rule: str = "count"
    label_ylim_fraction: float = 0.25  # used only when label_rule == "ylim_fraction"


@dataclass
class Paths:
    """All input root paths."""
    table_csv: Path
    af3_lig: Path
    af3_nolig: Path
    pdb_structures: Path
    pose_rmsds: Path
    threshold_table: Path            # CSV with 'af_pdb' and 'open' cols, used by find_rmsd


@dataclass
class PlotConfig:
    paths: Paths
    conditions: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    pose_cutoff: float = 2.0
    split_open: bool = True
    normalize: bool = False           # set True for equal-height bars (fractions)
    out_dir: Path = Path("plots_out")
    figure_name: str = "stacked_states"

    # Minimum raw count for a segment to get its number drawn. Segments at or
    # below this value are left unlabeled.
    label_min_count: int = 50

    # CSV column names — adjust here if your files differ
    col_model: str = "model"
    col_loop_rmsd: str = "loop_rmsd"
    col_pose_model: str = "pdb_seed"   # pose CSVs use 'pdb_seed'
    col_pose_rmsd: str = "lig_RMSD"    # pose CSVs use 'lig_RMSD'

    def state_specs(self) -> list:
        """State list (bottom-up stacking order) based on split_open."""
        if self.split_open:
            return [
                StateSpec("closed",         "Closed",                 COLOR_CLOSED),
                StateSpec("open_correct",   "Open (correct pose)",    COLOR_OPEN_CORRECT),
                StateSpec("open_incorrect", "Open (incorrect pose)",  COLOR_OPEN_INCORRECT),
                StateSpec("open",           "Open",                   COLOR_OPEN),
                StateSpec("other",          "Other",                  COLOR_OTHER),
            ]
        return [
            StateSpec("closed", "Closed", COLOR_CLOSED),
            StateSpec("open",   "Open",   COLOR_OPEN),
            StateSpec("other",  "Other",  COLOR_OTHER),
        ]


# ===========================================================================
# Folder resolvers — one per condition
# ===========================================================================

def resolve_af3_lig(pdb_lower, pdb_upper, paths):
    return paths.af3_lig / pdb_lower

def resolve_pdb_bound(pdb_lower, pdb_upper, paths):
    return paths.pdb_structures / f"{pdb_upper}_pdbs" / "bound"

def resolve_af3_nolig(pdb_lower, pdb_upper, paths):
    return paths.af3_nolig / pdb_lower

def resolve_pdb_unbound(pdb_lower, pdb_upper, paths):
    return paths.pdb_structures / f"{pdb_upper}_pdbs" / "unbound"


DEFAULT_CONDITIONS = [
    ConditionSpec("af3_lig",     "AF3 w/ Ligand",   resolve_af3_lig,     has_pose=True,
                  label_rule="count"),
    ConditionSpec("pdb_bound",   "PDB Bound",       resolve_pdb_bound,   has_pose=False,
                  label_rule="ylim_fraction", label_ylim_fraction=0.25),
    ConditionSpec("af3_nolig",   "AF3 w/o Ligand",  resolve_af3_nolig,   has_pose=False,
                  label_rule="count"),
    ConditionSpec("pdb_unbound", "PDB Unbound",     resolve_pdb_unbound, has_pose=False,
                  label_rule="ylim_fraction", label_ylim_fraction=0.25),
]


# ===========================================================================
# I/O helpers
# ===========================================================================

def parse_pdb_id(cell) -> Optional[str]:
    if cell is None:
        return None
    s = str(cell).strip()
    if not s:
        return None
    return s.split(".")[0].upper()


def _normalize_model_key(s: str) -> str:
    """
    Normalize a model identifier so loop-RMSD and pose-RMSD keys match.

    Strips common suffixes (.pdb, .cif), leading/trailing whitespace, and
    upper-cases everything. Returns the cleaned string.
    """
    if s is None:
        return ""
    t = str(s).strip()
    # Strip common extensions
    for ext in (".pdb", ".cif", ".pdb.gz"):
        if t.lower().endswith(ext):
            t = t[: -len(ext)]
    return t.upper()


def _read_two_col_csv(path: Path, model_col: str, rmsd_col: str,
                     model_out: str = "model", rmsd_out: str = "rmsd") -> Optional[pl.DataFrame]:
    """Generic reader for a CSV with a model-id column and a numeric column."""
    if not path.exists():
        return None
    try:
        df = pl.read_csv(path, infer_schema_length=10000)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)
        return None

    cols_lower = {c.lower().strip(): c for c in df.columns}
    mcol = cols_lower.get(model_col.lower())
    rcol = cols_lower.get(rmsd_col.lower())

    # Fall back: pick first column for model, first column containing 'rmsd' for value
    if mcol is None:
        mcol = df.columns[0]
    if rcol is None:
        rcands = [c for c in df.columns if "rmsd" in c.lower()]
        if not rcands:
            return None
        rcol = rcands[0]

    try:
        df = df.select([
            pl.col(mcol).cast(pl.Utf8).alias(model_out),
            pl.col(rcol).cast(pl.Float64, strict=False).alias(rmsd_out),
        ])
    except Exception:
        return None

    # Normalize model key: strip .pdb/.cif, uppercase, trim
    df = df.with_columns(
        pl.col(model_out)
          .str.strip_chars()
          .str.to_uppercase()
          .str.replace(r"\.PDB$", "")
          .str.replace(r"\.CIF$", "")
          .alias(model_out)
    )

    # Deduplicate by model (matches the original counting script)
    df = df.unique(subset=[model_out], keep="first")
    return df


def read_rmsd_csv(path: Path, cfg: 'PlotConfig') -> Optional[pl.DataFrame]:
    return _read_two_col_csv(path, cfg.col_model, cfg.col_loop_rmsd,
                             model_out="model", rmsd_out="rmsd")


def read_pose_rmsd(pdb_lower: str, cfg: 'PlotConfig') -> Optional[pl.DataFrame]:
    path = cfg.paths.pose_rmsds / f"{pdb_lower}_lig_RMSDs.csv"
    df = _read_two_col_csv(path, cfg.col_pose_model, cfg.col_pose_rmsd,
                           model_out="model", rmsd_out="pose_rmsd")
    if df is None:
        return None
    # Pose CSVs prefix keys with the pdb id (e.g. "1ALB_SEED-25_SAMPLE-2"),
    # while loop-RMSD CSVs use just "SEED-25_SAMPLE-2". Strip the prefix.
    prefix = f"{pdb_lower.upper()}_"
    df = df.with_columns(
        pl.when(pl.col("model").str.starts_with(prefix))
          .then(pl.col("model").str.slice(len(prefix)))
          .otherwise(pl.col("model"))
          .alias("model")
    )
    return df


# ===========================================================================
# Threshold lookup — replicates find_rmsd(PDB_ID) / 2
# ===========================================================================

def find_threshold_for_pdb(pdb_upper: str, cfg: 'PlotConfig') -> Optional[float]:
    table_path = cfg.paths.threshold_table
    if not table_path.exists():
        print(f"[WARN] threshold table not found: {table_path}", file=sys.stderr)
        return None

    try:
        df = pl.read_csv(table_path, infer_schema_length=10000)
    except Exception:
        return None

    if "af_pdb" not in df.columns or "open" not in df.columns:
        return None

    match = df.filter(
        pl.col("af_pdb").cast(pl.Utf8).str.slice(0, 4).str.to_uppercase()
        == pdb_upper[:4].upper()
    )
    if match.height == 0:
        return None

    open_pdb = match.select("open").item(0, 0)
    if open_pdb is None:
        return None
    open_pdb = str(open_pdb).strip().upper()[:4]

    # 3IXJ special case from the original script
    state = "unbound" if pdb_upper == "3IXJ" else "bound"
    csv_path = cfg.paths.pdb_structures / f"{pdb_upper}_pdbs" / state / "closed_rmsd.csv"
    rmsd_df = read_rmsd_csv(csv_path, cfg)
    if rmsd_df is None:
        return None

    hit = rmsd_df.filter(pl.col("model").str.slice(0, 4) == open_pdb[:4])
    if hit.height == 0:
        return None

    val = hit.select("rmsd").item(0, 0)
    if val is None:
        return None
    return float(val) / 2.0


# ===========================================================================
# Per-protein, per-condition state counting
# ===========================================================================

def count_states_for_condition(pdb_upper: str, condition: ConditionSpec,
                               threshold: float, cfg: PlotConfig) -> dict:
    """
    Classify every model in this (protein, condition) by closed/open/other,
    optionally splitting open by pose RMSD. Returns a dict with one int per
    canonical state name in cfg.state_specs().
    """
    pdb_lower = pdb_upper.lower()
    folder = condition.folder_resolver(pdb_lower, pdb_upper, cfg.paths)

    closed_df = read_rmsd_csv(folder / "closed_rmsd.csv", cfg)
    open_df   = read_rmsd_csv(folder / "open_rmsd.csv",   cfg)

    state_names = [s.name for s in cfg.state_specs()]
    counts = {s: 0 for s in state_names}

    if closed_df is None or open_df is None:
        return counts

    joined = closed_df.rename({"rmsd": "rmsd_closed"}).join(
        open_df.rename({"rmsd": "rmsd_open"}),
        on="model", how="full", coalesce=True,
    )

    classified = joined.with_columns(
        pl.when(pl.col("rmsd_closed") <= threshold).then(pl.lit("closed"))
        .when(pl.col("rmsd_open")   <= threshold).then(pl.lit("open"))
        .otherwise(pl.lit("other"))
        .alias("state")
    )

    should_split = (
        cfg.split_open
        and condition.has_pose
        and "open_correct" in counts
        and "open_incorrect" in counts
    )

    pose_df = read_pose_rmsd(pdb_lower, cfg) if should_split else None

    if should_split and pose_df is not None:
        classified = classified.join(pose_df, on="model", how="left")
        classified = classified.with_columns(
            pl.when(pl.col("state") == "open")
              .then(
                  pl.when(pl.col("pose_rmsd").is_null())
                    .then(pl.lit("open"))  # no pose data -> plain open
                  .when(pl.col("pose_rmsd") <= cfg.pose_cutoff)
                    .then(pl.lit("open_correct"))
                  .otherwise(pl.lit("open_incorrect"))
              )
              .otherwise(pl.col("state"))
              .alias("state_final")
        )
        state_col = "state_final"
    else:
        state_col = "state"

    agg = classified.group_by(state_col).len()
    for row in agg.iter_rows(named=True):
        name = row[state_col]
        if name in counts:
            counts[name] = int(row["len"])
    return counts


# ===========================================================================
# Plotting primitives
# ===========================================================================

def _style_ax(ax, title, show_ylabel, ylabel):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", direction="out", length=3, width=0.8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4, linewidth=0.6, color="#888888")
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=18, loc="center")
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


def annotate_missing(ax, msg="No data"):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            fontsize=9, color="#888888", style="italic", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _luminance(hex_color: str) -> float:
    """Approximate relative luminance — used to pick a readable text color."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return 0.299 * r + 0.587 * g + 0.114 * b


def _text_color_for(bg_hex: str) -> str:
    return "#111111" if _luminance(bg_hex) > 0.55 else "#ffffff"


def draw_stacked_bars(ax, protein_labels, counts_per_protein, state_specs,
                      normalize, show_ylabel, ylabel, title,
                      label_rule: str = "count",
                      label_min_count: int = 50,
                      label_ylim_fraction: float = 0.25,
                      max_proteins: Optional[int] = None):
    """
    Draw stacked bars. Each non-zero segment may be annotated with its RAW
    count, according to `label_rule`:

      "count"         -> label only segments whose raw count > label_min_count.
      "ylim_fraction" -> label only segments whose plotted height exceeds
                         label_ylim_fraction of the axis y-range.

    There is no promote-to-total fallback.

    max_proteins: if set, the x-axis is sized for this many slots regardless
    of how many proteins are actually drawn. This keeps the physical bar
    WIDTH constant across rows that have different protein counts. Bars are
    left-aligned and the right side of the axis is left blank.
    """
    if not protein_labels:
        annotate_missing(ax)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        # Even when empty, keep the axis range consistent
        if max_proteins:
            ax.set_xlim(-0.6, max_proteins - 0.4)
        return

    x = np.arange(len(protein_labels))
    bottoms = np.zeros(len(protein_labels), dtype=float)

    totals = np.array([sum(c.values()) for c in counts_per_protein], dtype=float)
    denom = np.where(totals > 0, totals, 1.0)

    # Draw all bars and keep the rectangles so we can label them afterwards.
    bar_rects_by_state = {}
    for spec in state_specs:
        raw_vals = np.array([c.get(spec.name, 0) for c in counts_per_protein], dtype=float)
        fractions = raw_vals / denom
        plot_vals = fractions if normalize else raw_vals

        bars = ax.bar(x, plot_vals, bottom=bottoms, width=0.7,
                      color=spec.color, edgecolor="#333333", linewidth=0.4,
                      label=spec.label)
        bar_rects_by_state[spec.name] = (bars, raw_vals, fractions)
        bottoms += plot_vals

    ax.set_xticks(x)
    ax.set_xticklabels(protein_labels, rotation=40, ha="right", fontsize=9)
    _style_ax(ax, title, show_ylabel, ylabel)

    # Lock x-axis range so bar width is identical across rows with different
    # numbers of proteins.
    if max_proteins:
        ax.set_xlim(-0.6, max_proteins - 0.4)

    if normalize:
        ax.set_ylim(0, 1.02)

    # ---- Labeling pass ----
    # Decide, per segment, whether to draw its raw count, based on label_rule.
    #   "count"         -> raw count > label_min_count
    #   "ylim_fraction" -> plotted segment height > label_ylim_fraction * y-range
    y_lo, y_hi = ax.get_ylim()
    y_range = max(y_hi - y_lo, 1e-9)
    min_segment_height = label_ylim_fraction * y_range

    for spec in state_specs:
        bars, raw_vals, fractions = bar_rects_by_state[spec.name]
        txt_color = _text_color_for(spec.color)
        for i in range(len(protein_labels)):
            count = int(raw_vals[i])
            if count == 0:
                continue
            rect = bars[i]

            if label_rule == "ylim_fraction":
                if rect.get_height() <= min_segment_height:
                    continue
            else:  # "count"
                if count <= label_min_count:
                    continue

            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_y() + rect.get_height() / 2,
                str(count),
                ha="center", va="center",
                fontsize=8, fontweight="bold", color=txt_color,
            )


# ===========================================================================
# Top-level figure builder
# ===========================================================================

def build_protein_list_for_group(df: pl.DataFrame, group: str) -> list:
    rows = []
    for r in df.filter(pl.col("group") == group).iter_rows(named=True):
        pid = parse_pdb_id(r["af_pdb"])
        if pid:
            rows.append({"pdb_id": pid, "name": r["name"]})
    return rows


def make_figure(cfg: PlotConfig):
    if not cfg.paths.table_csv.exists():
        raise FileNotFoundError(f"Table CSV not found: {cfg.paths.table_csv}")

    # Sanity check: warn loudly if any expected root directory is missing.
    # Missing dirs are the most common cause of all-zero figures.
    expected = {
        "af3_lig":         cfg.paths.af3_lig,
        "af3_nolig":       cfg.paths.af3_nolig,
        "pdb_structures":  cfg.paths.pdb_structures,
        "pose_rmsds":      cfg.paths.pose_rmsds,
        "threshold_table": cfg.paths.threshold_table,
    }
    missing = {k: v for k, v in expected.items() if not v.exists()}
    if missing:
        print("=" * 70, file=sys.stderr)
        print("[WARN] The following input paths DO NOT EXIST:", file=sys.stderr)
        for k, v in missing.items():
            print(f"   {k:18s} -> {v}", file=sys.stderr)
        print("Continuing, but bars for affected proteins/conditions will be empty.",
              file=sys.stderr)
        print("=" * 70, file=sys.stderr)

    df = pl.read_csv(cfg.paths.table_csv, infer_schema_length=10000)
    for c in ("af_pdb", "group", "name"):
        if c not in df.columns:
            cands = [x for x in df.columns if x.lower() == c]
            if not cands:
                raise ValueError(f"Missing column '{c}' in {cfg.paths.table_csv}")
            df = df.rename({cands[0]: c})
    df = df.with_columns(pl.col("group").cast(pl.Utf8))

    state_specs = cfg.state_specs()
    n_rows = len(cfg.groups)
    n_cols = len(cfg.conditions)
    if n_rows == 0 or n_cols == 0:
        print("[ERROR] No groups or no conditions configured.", file=sys.stderr)
        return

    fig_w = max(4.0 * n_cols, 12)
    fig_h = max(3.0 * n_rows + 1.0, 4.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=300, squeeze=False)
    fig.subplots_adjust(wspace=0.18, hspace=0.45,
                        left=0.09, right=0.97,
                        top=0.92, bottom=0.08)

    ylabel = "Fraction of models" if cfg.normalize else "Count"

    # Threshold cache so we only compute per pdb once
    threshold_cache: dict = {}

    def get_threshold(pdb_upper):
        if pdb_upper not in threshold_cache:
            threshold_cache[pdb_upper] = find_threshold_for_pdb(pdb_upper, cfg)
        return threshold_cache[pdb_upper]

    # Pre-pass: how many proteins per group? Use the maximum to size every
    # subplot's x-axis so bar widths stay equal across rows.
    proteins_by_group = {g: build_protein_list_for_group(df, g) for g in cfg.groups}
    max_proteins = max((len(v) for v in proteins_by_group.values()), default=1)
    max_proteins = max(max_proteins, 1)

    for r, group in enumerate(cfg.groups):
        proteins = proteins_by_group[group]

        # Row label (group) on the left
        axes[r][0].annotate(
            f"Group {group}",
            xy=(-0.25, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center",
            fontsize=13, fontweight="bold",
        )

        if not proteins:
            for c in range(n_cols):
                title = cfg.conditions[c].title if r == 0 else ""
                annotate_missing(axes[r][c], f"No proteins")
                if title:
                    axes[r][c].set_title(title, fontsize=12, fontweight="bold", pad=6)
                axes[r][c].set_xlim(-0.6, max_proteins - 0.4)
            print(f"[WARN] No proteins found for group {group}")
            continue

        labels = [PROTEIN_LABELS.get(p["name"], p["name"]) for p in proteins]

        for c, cond in enumerate(cfg.conditions):
            counts_list = []
            for p in proteins:
                t = get_threshold(p["pdb_id"])
                if t is None:
                    counts_list.append({s.name: 0 for s in state_specs})
                    continue
                counts_list.append(count_states_for_condition(p["pdb_id"], cond, t, cfg))

            draw_stacked_bars(
                axes[r][c],
                protein_labels=labels,
                counts_per_protein=counts_list,
                state_specs=state_specs,
                normalize=cfg.normalize,
                show_ylabel=(c == 0),
                ylabel=ylabel,
                title=cond.title if r == 0 else "",
                label_rule=cond.label_rule,
                label_min_count=cfg.label_min_count,
                label_ylim_fraction=cond.label_ylim_fraction,
                max_proteins=max_proteins,
            )

    # Single legend at bottom. When split_open=True, suppress the plain "Open"
    # entry — pose-bearing conditions show the correct/incorrect breakdown,
    # and non-pose conditions still draw plain Open in the same color as
    # "Open (correct pose)", so the legend swatch covers both visually.
    legend_specs = state_specs
    if cfg.split_open:
        legend_specs = [s for s in state_specs if s.name != "open"]

    legend_handles = [
        mpatches.Patch(facecolor=s.color, edgecolor="#333333", linewidth=0.4, label=s.label)
        for s in legend_specs
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.12),
        frameon=True, framealpha=0.92, edgecolor="#888888",
        fontsize=9,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(cfg.out_dir / f"{cfg.figure_name}.{ext}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {cfg.out_dir / cfg.figure_name}.png/.pdf")


# ===========================================================================
# Entry point — edit paths/groups/toggles here
# ===========================================================================

def main():
    paths = Paths(
        table_csv       = Path("pnas_table_mod.csv"),
        af3_lig         = Path("pnas_af3_lig"),
        af3_nolig       = Path("pnas_af3_nolig"),
        pdb_structures  = Path("pdb_structures_final"),
        pose_rmsds      = Path("pose_rmsds"),
        threshold_table = Path("pnas_table_mod.csv"),
    )

    # ----- Path set B: "unbiased" / no_cutoff (swap to this if needed) -----
    # paths = Paths(
    #     table_csv       = Path("unbiased_table_cut.csv"),
    #     af3_lig         = Path("af3_lig_unbiased"),
    #     af3_nolig       = Path("af3_nolig_unbiased"),
    #     pdb_structures  = Path("pdb_structures_no_cutoff"),
    #     pose_rmsds      = Path("pose_rmsds"),
    #     threshold_table = Path("pnas_table_mod.csv"),
    # )

    # # --- Main figure: all groups, all conditions, with open-pose split ---
    cfg = PlotConfig(
        paths          = paths,
        conditions     = DEFAULT_CONDITIONS,
        groups         = ["3"],
        pose_cutoff    = 2.0,
        split_open     = True,
        normalize      = False,        # raw counts; bar widths fixed across rows
        label_min_count = 50,          # only show numbers for segments > 50
        out_dir        = Path("plots_out"),
        figure_name    = "stacked_states_all_groups_grp3",
    )
    make_figure(cfg)

    # --- Example variants (uncomment as needed) ---

    # AF3-only (drop PDB columns):
    # make_figure(PlotConfig(
    #     paths           = paths,
    #     conditions      = [c for c in DEFAULT_CONDITIONS if c.key.startswith("af3")],
    #     groups          = ["1a", "1b", "1c", "2", "3"],
    #     split_open      = True,
    #     label_min_count = 50,
    #     out_dir         = Path("plots_out"),
    #     figure_name     = "stacked_states_af3_only",
    # ))
    #
    # No pose split, normalized fractions:
    # make_figure(PlotConfig(
    #     paths           = paths,
    #     conditions      = [c for c in DEFAULT_CONDITIONS if c.key.startswith("af3")],
    #     groups          = ["1a", "1b", "1c", "2", "3"],
    #     split_open      = False,
    #     normalize       = False,
    #     label_min_count = 50,            # only show numbers for segments > 50
    #     out_dir         = Path("plots_out"),
    #     figure_name     = "stacked_states_no_pose_split_af3_only",
    # ))


if __name__ == "__main__":
    main()