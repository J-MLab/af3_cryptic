"""
-script to find the ligand RMSDs of AF3 predicted w/ ligand and bound reference structure
Author: Felix Tuchscherer
Date: June 2025
"""

import os
import pandas as pd
import numpy as np
from utils.lig_rmsd_funct import *

# helper function for getting the bound reference structure of the predicted AF models
df = pd.read_csv("pnas_table_mod.csv")


def get_bound_by_af_prefix(af_prefix, table):
    """
    Given the first 4 chars of an AF PDB code (e.g. "1GX8"),
    return the corresponding 'bound' PDB ID from the table.
    If no match is found, returns None. If multiple matches,
    prints them and returns a list of bound IDs.
    """
    mask = table["af_pdb"].str[:4] == af_prefix
    matches = table.loc[mask, ["af_pdb", "bound"]]

    if matches.empty:
        print(f"No entries found with af_pdb starting '{af_prefix}'")
        return None

    if len(matches) > 1:
        print(f"Multiple matches for prefix '{af_prefix}':")
        print(matches.to_string(index=False))
        return matches["bound"].tolist()

    return matches.iloc[0]["bound"]


failed_cases = []
base_path = "pnas_af3_lig"
out_dir = "pose_rmsds"
os.makedirs(out_dir, exist_ok=True)

for pdb in os.listdir(base_path):  # iterate through each PDB
    if pdb not in ["1alb", "1yes", "1rhb"]:
        continue
    pdb_path = os.path.join(base_path, pdb)

    # — create a fresh dict for this PDB, e.g. “1alb_lig_RMSDs = {}”
    dict_name = f"{pdb}_lig_RMSDs"
    globals()[dict_name] = {}

    # get the bound ref_structure
    bound_ref = get_bound_by_af_prefix(pdb.upper(), df)
    ref_model_path = f"bound_ref_structs/{bound_ref}.cif"

    print(f"finding ligand RMSD for {pdb}")
    for seed in os.listdir(pdb_path):  # iterate through each seed
        if os.path.isdir(pdb_path) and seed.startswith("seed"):
            seed_path = os.path.join(pdb_path, seed)
            for model in os.listdir(seed_path):
                if model == "model.cif":
                    pred_model_path = os.path.join(seed_path, model)
                    try:
                        rmsd = calculate_pose_rmsd(
                            ref_model_path, pred_model_path
                        )  # all the calculations done here
                        # — store into our per‐PDB dictionary
                        globals()[dict_name][f"{pdb}_{seed}"] = rmsd
                    except Exception as e:
                        print(f"[SKIPPED] Failed for {pdb}_{seed}: {e}")
                        failed_cases.append(
                            {"pdb_seed": f"{pdb}_{seed}", "error": str(e)}
                        )
                        continue

    # output as a .csv using pandas, pulling from the per-PDB dict
    rmsd_df = pd.DataFrame(
        list(globals()[dict_name].items()), columns=["pdb_seed", "lig_RMSD"]
    )
    out_csv = os.path.join(out_dir, f"{pdb}_lig_RMSDs.csv")
    rmsd_df.to_csv(out_csv, index=False)
    print(f"[OK] wrote: {out_csv}")

if failed_cases:
    error_df = pd.DataFrame(failed_cases)
    error_csv = os.path.join(out_dir, "error_ligands.csv")
    error_df.to_csv(error_csv, index=False)
    print(f"[INFO] Wrote error cases to: {error_csv}")
else:
    print("[INFO] No errors encountered.")
