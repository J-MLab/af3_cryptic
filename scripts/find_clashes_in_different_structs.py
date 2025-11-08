"""
Script that finds clashes between ligands and protein residues across different structures.
Author: Maria Lazou
September 2025
"""

import os
import numpy as np
from Bio import PDB
from Bio.PDB.Polypeptide import is_aa
from collections import defaultdict
from pathlib import Path
import polars as pl


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


def count_clashes(ref_structure, test_structure, residues, clash_cutoff=0.63):
    clash_cutoffs = {
        i + "_" + j: (clash_cutoff * (atom_radii[i] + atom_radii[j]))
        for i in atom_radii
        for j in atom_radii
    }

    protein_atoms = [
        x
        for x in ref_structure.get_atoms()
        if x.element in atom_radii
        and _is_protein_residue(x.get_parent())
        and x.get_parent().get_id()[1] in residues
    ]

    protein_coords = np.array([a.coord for a in protein_atoms], dtype="d")

    ligand_atoms = [
        x
        for x in test_structure.get_atoms()
        if x.element in atom_radii and _is_ligand_residue(x.get_parent())
    ]
    ligand_coords = np.array([a.coord for a in ligand_atoms], dtype="d")

    if len(protein_atoms) == 0 or len(ligand_atoms) == 0:
        print("No atoms")
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
    print(unique_residues_per_ligand(clashes_dict))
    return unique_residues_per_ligand(clashes_dict)
