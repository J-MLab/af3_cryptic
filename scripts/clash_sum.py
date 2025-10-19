"""
Main script that outputs the summary csv of the number of clashes for the 2 cases
"""

import os
import numpy as np
from Bio import PDB
from Bio.PDB.Polypeptide import is_aa
from collections import defaultdict
from pathlib import Path
import polars as pl

# === SETTINGS ===
rmsd_folder = Path("pose_rmsds")  # contains *_2.0.csv and *_2.5.csv
cif_root = Path("pnas_af3_lig")  # root folder containing CIFs
output_file_2 = "clash_summary_2.csv"
output_file_25 = "clash_summary_25.csv"

# === Clash detection code (from your script) ===
parser = PDB.MMCIFParser(QUIET=True)

atom_radii = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "F": 1.47,
    "P": 1.80,
    "CL": 1.75,
    "MG": 1.73,
}
_WATER = {"HOH", "WAT", "DOD", "H2O"}


def _is_protein_residue(res) -> bool:
    return res.id[0] == " " and is_aa(res, standard=True)


def _is_ligand_residue(res, ligand_resnames=None, ignore_waters=True):
    resname = res.resname.strip()
    hetflag = res.id[0]
    if ligand_resnames is not None:
        return resname in ligand_resnames
    if hetflag != " ":
        if ignore_waters and resname in _WATER:
            return False
        return True
    return False


def unique_residues_per_ligand(clashes_dict):
    per_lig = defaultdict(set)
    for (lig_id, _), hits in clashes_dict.items():
        per_lig[lig_id].update(resid[1] for _, resid, _ in hits)
    return {lig: sorted(v) for lig, v in per_lig.items()}


def count_clashes(structure, clash_cutoff=0.63):
    clash_cutoffs = {
        i + "_" + j: (clash_cutoff * (atom_radii[i] + atom_radii[j]))
        for i in atom_radii
        for j in atom_radii
    }
    protein_atoms = [
        x
        for x in structure.get_atoms()
        if x.element in atom_radii and _is_protein_residue(x.get_parent())
    ]
    protein_coords = np.array([a.coord for a in protein_atoms], dtype="d")
    ligand_atoms = [
        x
        for x in structure.get_atoms()
        if x.element in atom_radii and _is_ligand_residue(x.get_parent())
    ]
    ligand_coords = np.array([a.coord for a in ligand_atoms], dtype="d")

    if len(protein_atoms) == 0 or len(ligand_atoms) == 0:
        return {}

    kdt = PDB.kdtrees.KDTree(protein_coords)
    clashes_dict = {}
    for lig_atm in ligand_atoms:
        thresholds = [
            (k, v) for k, v in clash_cutoffs.items() if k[0] == lig_atm.element
        ]
        for pair, t in thresholds:
            kdt_search = kdt.search(np.array(lig_atm.coord, dtype="d"), t)
            clashes = [
                (
                    protein_atoms[a.index].get_serial_number(),
                    protein_atoms[a.index].get_parent().get_id(),
                    protein_atoms[a.index].element,
                )
                for a in kdt_search
                if protein_atoms[a.index].element == pair[-1]
            ]
            if len(clashes) > 0:
                clashes_dict[(lig_atm.get_parent().get_id()[0], lig_atm.get_id())] = (
                    clashes
                )

    return unique_residues_per_ligand(clashes_dict)


# === MAIN SCRIPT ===
correct_pose = {"pdb_id": [], "seed": [], "criteria": [], "num_clashes": []}
near_correst_poses = {"pdb_id": [], "seed": [], "criteria": [], "num_clashes": []}

for i, filename in enumerate(rmsd_folder.iterdir()):
    print(filename.stem)
    if filename.name.endswith("_2.0.csv") or filename.name.endswith("_2.5.csv"):
        case = "2.0" if filename.stem.endswith("_2.0") else "2.5"
        parent_pdb = filename.stem.split("_")[0]  # first part before "_lig_RMSDs"

        df = pl.read_csv(filename)
        clash_count = 0
        total_rows = 0

        for row in df.iter_rows(named=True):
            row_pdb = str(row["pdb_seed"])[
                5:
            ]  # protein ID in the row, slice at 5 to skip the pdbID
            cif_path = cif_root / Path(parent_pdb, row_pdb, "model.cif")

            if not os.path.exists(cif_path):
                print(f"Missing CIF: {cif_path}, skipping")
                continue

            total_rows += 1
            try:
                structure = parser.get_structure(row_pdb, cif_path)
                clashes = count_clashes(structure)
                if clashes:  # dictionary is non-empty
                    print(f"Clashes found in ", parent_pdb)
                    if case == "2.0":
                        correct_pose["pdb_id"].append(parent_pdb)
                        correct_pose["seed"].append(row_pdb)
                        correct_pose["criteria"].append(r"< 2.0 Å")
                        correct_pose["num_clashes"].append(
                            len(list(clashes.values())[0])
                        )
                    else:
                        near_correst_poses["pdb_id"].append(parent_pdb)
                        near_correst_poses["seed"].append(row_pdb)
                        near_correst_poses["criteria"].append("< 2.5 Å")
                        near_correst_poses["num_clashes"].append(
                            len(list(clashes.values())[0])
                        )

            except Exception as e:
                print(f"Error parsing {cif_path}: {e}")


correct_pose_df = pl.DataFrame(correct_pose)
near_correst_poses_df = pl.DataFrame(near_correst_poses)

correct_pose_df.write_csv(output_file_2)
near_correst_poses_df.write_csv(output_file_25)
