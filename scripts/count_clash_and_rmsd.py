"""Script to count the number of models in each state (open, closed, neither) based on clash and RMSD thresholds.
Saves the results to a CSV file.
Author: Maria Lazou
October 2025"""

import polars as pl
from pathlib import Path

pdb_directory = Path("pdb_structures_final/")
af3_lig_directory = Path("pnas_af3_lig")
af3_nolig_directory = Path("pnas_af3_nolig")
clash_counts = Path("count_by_clash")

state_counts = pl.read_csv("total_state_counts_mod.csv")

results = {"State": [], "Count": []}

for dir in pdb_directory.iterdir():
    pdb = dir.stem.split("_")[0]
    rmsd_t = (
        state_counts.filter(pl.col("State") == "rmsd/2_" + pdb).select("Count").item()
    )
    for state_dir in dir.iterdir():
        if state_dir.is_dir():
            count_dict = {"open": 0, "closed": 0, "neither": 0}
            clash_df = pl.read_csv(
                Path(clash_counts, f"{pdb}_pdbs_PDB_{state_dir.stem}.csv")
            )
            open_df = pl.read_csv(Path(state_dir, "open_rmsd.csv")).rename(
                {"loop_rmsd": "open"}
            )
            closed_df = pl.read_csv(Path(state_dir, "closed_rmsd.csv")).rename(
                {"loop_rmsd": "closed"}
            )

            clash_df = clash_df.join(
                open_df, on="model", how="outer", coalesce=True
            ).join(
                closed_df, on="model", how="outer", coalesce=True
            )  # ERROR 3: 'closed' should be 'closed_df'
            clash_df = clash_df.drop_nulls()
            for row in clash_df.iter_rows(named=True):
                if row["closed"] < rmsd_t or row["clash"] == "yes":
                    count_dict["closed"] += 1
                elif row["open"] < rmsd_t and row["clash"] != "yes":
                    count_dict["open"] += 1
                elif (
                    row["closed"] > rmsd_t
                    and row["open"] > rmsd_t
                    and row["clash"] != "yes"
                ):
                    count_dict["neither"] += 1

            for key in count_dict.keys():
                results["State"].append(f"pdb_{state_dir.stem}_{pdb}_{key}")
                results["Count"].append(count_dict[key])

for dir in af3_nolig_directory.iterdir():
    pdb = dir.stem.upper()
    rmsd_t = (
        state_counts.filter(pl.col("State") == "rmsd/2_" + pdb).select("Count").item()
    )
    if dir.is_dir():
        count_dict = {"open": 0, "closed": 0, "neither": 0}
        clash_df = pl.read_csv(Path(clash_counts, f"{pdb.lower()}_af3_nolig.csv"))
        open_df = pl.read_csv(Path(dir, "open_rmsd.csv")).rename({"loop_rmsd": "open"})
        closed_df = pl.read_csv(Path(dir, "closed_rmsd.csv")).rename(
            {"loop_rmsd": "closed"}
        )

        clash_df = clash_df.join(open_df, on="model", how="outer", coalesce=True).join(
            closed_df, on="model", how="outer", coalesce=True
        )  # ERROR 3: 'closed' should be 'closed_df'
        clash_df = clash_df.drop_nulls()
        for row in clash_df.iter_rows(named=True):
            if row["closed"] < rmsd_t or row["clash"] == "yes":
                count_dict["closed"] += 1
            elif row["open"] < rmsd_t and row["clash"] != "yes":
                count_dict["open"] += 1
            elif (
                row["closed"] > rmsd_t
                and row["open"] > rmsd_t
                and row["clash"] != "yes"
            ):
                count_dict["neither"] += 1

        for key in count_dict.keys():
            results["State"].append(f"af3_unbound_{pdb}_{key}")
            results["Count"].append(count_dict[key])

for dir in af3_lig_directory.iterdir():
    pdb = dir.stem.upper()
    rmsd_t = (
        state_counts.filter(pl.col("State") == "rmsd/2_" + pdb).select("Count").item()
    )
    if dir.is_dir():
        count_dict = {"open": 0, "closed": 0, "neither": 0}
        clash_df = pl.read_csv(Path(clash_counts, f"{pdb.lower()}_af3_lig.csv"))
        open_df = pl.read_csv(Path(dir, "open_rmsd.csv")).rename({"loop_rmsd": "open"})
        closed_df = pl.read_csv(Path(dir, "closed_rmsd.csv")).rename(
            {"loop_rmsd": "closed"}
        )
        clash_df = clash_df.join(open_df, on="model", how="outer", coalesce=True).join(
            closed_df, on="model", how="outer", coalesce=True
        )
        clash_df = clash_df.drop_nulls()
        for row in clash_df.iter_rows(named=True):
            if row["closed"] < rmsd_t or row["clash"] == "yes":
                count_dict["closed"] += 1
            elif row["open"] < rmsd_t and row["clash"] != "yes":
                count_dict["open"] += 1
            elif (
                row["closed"] > rmsd_t
                and row["open"] > rmsd_t
                and row["clash"] != "yes"
            ):
                count_dict["neither"] += 1

        for key in count_dict.keys():
            results["State"].append(f"af3_bound_{pdb}_{key}")
            results["Count"].append(count_dict[key])

results_df = pl.DataFrame(results)
results_df.write_csv("total_state_counts_by_clash.csv")
