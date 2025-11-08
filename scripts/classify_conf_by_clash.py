"""This script classifies AlphaFold3 predicted structures based on steric clashes
with ligands from reference crystal structures.
Author: Maria Lazou
October 2025"""

import os
import pymol
from pymol import cmd
from Bio.PDB import MMCIFParser, is_aa
import numpy as np
import pandas as pd
import shutil
from scripts.find_clashes_in_different_structs import count_clashes
from Bio import PDB
from pathlib import Path
import polars as pl
from scipy.spatial.distance import cdist
from Bio.PDB.Polypeptide import three_to_index


def safe_three_to_one(resname):
    """Convert three-letter amino acid code to one-letter code."""
    try:
        return three_to_index(resname)
    except KeyError:
        return None


def map_residues_by_coordinates(ref_chain, mobile_chain, threshold=10):
    """
    Map residues between two aligned chains based on Cα atom proximity.

    Args:
        ref_chain (Chain): Chain from the reference structure (e.g., crystal).
        mobile_chain (Chain): Chain from the aligned structure.
        threshold (float): Distance cutoff (Å) for matching Cα atoms.

    Returns:
        dict: Mapping of ref_residue (key) -> mobile_residue (value).
    """
    # Filter for amino acid residues
    ref_residues = [res for res in ref_chain if is_aa(res)]
    mobile_residues = [res for res in mobile_chain if is_aa(res)]

    # Extract Cα atom coordinates and residue types
    ref_coords = [
        (res, res["CA"].get_coord(), safe_three_to_one(res.get_resname()))
        for res in ref_residues
        if "CA" in res and safe_three_to_one(res.get_resname()) is not None
    ]
    mobile_coords = [
        (res, res["CA"].get_coord(), safe_three_to_one(res.get_resname()))
        for res in mobile_residues
        if "CA" in res and safe_three_to_one(res.get_resname()) is not None
    ]

    ref_residues_list, ref_calpha, ref_types = zip(*ref_coords)
    mobile_residues_list, mobile_calpha, mobile_types = zip(*mobile_coords)

    # Compute pairwise distances
    distances = cdist(ref_calpha, mobile_calpha, "euclidean")

    # Map residues based on type and minimum distance within the threshold
    residue_map = {}
    for i, (ref_res, ref_type) in enumerate(zip(ref_residues_list, ref_types)):
        matching_indices = [
            j for j, mob_type in enumerate(mobile_types) if mob_type == ref_type
        ]

        if matching_indices:
            matching_distances = distances[i, matching_indices]
            min_idx_in_matches = matching_distances.argmin()
            min_idx = matching_indices[min_idx_in_matches]

            if distances[i, min_idx] <= threshold:
                # Map residue numbers instead of objects
                ref_resnum = ref_res.get_id()[1]
                mobile_resnum = mobile_residues_list[min_idx].get_id()[1]
                residue_map[ref_resnum] = mobile_resnum

    return residue_map


parser = PDB.MMCIFParser(QUIET=True)


def parse_entry(entry: str):
    parts = [p.strip() for p in entry.replace(" ", "").split(",")]
    result = []
    for part in parts:
        if "-" in part:
            start, end = part.split("-")
            # remove 'arg' or 'syn' and convert to int
            start_num = int("".join(filter(str.isdigit, start)))
            end_num = int("".join(filter(str.isdigit, end)))
            result.extend(list(range(start_num, end_num + 1)))
        else:
            num = int("".join(filter(str.isdigit, part)))
            result.append(num)
    return result


def get_ligand_instances(structure_file, ligand_resn):
    """Extracts individual ligand instances and their geometric centers."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", structure_file)
    ligands = []  # Store individual ligand instances
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() == ligand_resn:
                    ligand_atoms = [
                        atom.coord for atom in residue if atom.element != "H"
                    ]
                    if ligand_atoms:
                        center = np.mean(ligand_atoms, axis=0)
                        ligands.append(
                            (center, ligand_atoms)
                        )  # Store center & full atom list
    return ligands


def compute_ligand_size(ligand_atoms):
    """Calculates the spatial extent of a ligand based on max pairwise atom distance."""
    max_distance = 0
    for i, atom1 in enumerate(ligand_atoms):
        for j, atom2 in enumerate(ligand_atoms):
            if i < j:
                distance = np.linalg.norm(atom1 - atom2)
                max_distance = max(max_distance, distance)
    return max_distance


def get_potential_ligands(structure_file):
    """Identify non-protein, non-nucleic residues as potential ligands."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("test", structure_file)
    n = {}
    ligands = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    ligand_resn = residue.get_resname().strip()
                    if ligand_resn == "HOH":
                        continue
                    atoms = [atom.coord for atom in residue if atom.element != "H"]
                    if atoms:
                        center = np.mean(
                            atoms, axis=0
                        )  # returns x,y,z coordinate of the center
                        if ligands.get(ligand_resn, "None") == "None":
                            ligands[ligand_resn] = (center, atoms)
                            n[ligand_resn] = 1
                        else:
                            n[ligand_resn] = n[ligand_resn] + 1
                            ligands[ligand_resn + "_" + str(n[ligand_resn])] = (
                                center,
                                atoms,
                            )
    return ligands, n


def check_ligand_binding(
    test_file, ref_ligands, threshold=5.0, scale_factor=0.5, overlap_threshold=2.0
):
    """Check if any ligand in the test structure is at the reference binding site."""
    print(f"Loading file: {test_file}")
    CRYSTAL_ADDITIVES = {"HOH", "DOD", "WAT"}
    cmd.load(test_file, "test")
    cmd.align("test", "ref")
    cmd.save(test_file, "test")
    test_ligands, quantity = get_potential_ligands(test_file)
    if not test_ligands:
        print("No potential ligands found in test structure.")
        return False
    for ref_center, ref_atoms in ref_ligands:
        for (
            test_ligand_ID,
            (test_center, test_atoms),
        ) in (
            test_ligands.items()
        ):  # made change here to correctly iterate over value not key
            if test_ligand_ID.split("_")[0] in CRYSTAL_ADDITIVES:
                continue
            # if quantity[test_ligand_ID.split('_')[0]] > 2:\
            #   continue
            distance = np.linalg.norm(ref_center - test_center)
            print(f"Checking ligand: Distance = {distance:.2f} Å")
            if (
                distance < threshold
            ):  # checks if the distance calculation is close enough
                print(f"Ligand is bound at the same site!")
                return True
                # Compare each atom of the reference ligand with each atom of the test ligand. In essence, I'm checking if any atom
                # in either ligand is within 2 angstroms of each other, if they are I'm counting this as a ligand overlap, and
                # will therefore add it to the bound list
            else:
                for ref_atom in ref_atoms:
                    for test_atom in test_atoms:
                        atom_distance = np.linalg.norm(ref_atom - test_atom)
                        if atom_distance < overlap_threshold:
                            print(
                                f"Ligands overlap! (Atom distance: {atom_distance:.2f} Ã…)"
                            )
                            return True
    print(
        "No ligand found at the reference binding site OR overlaps with the reference ligand"
    )
    return False


# Example usage
data = pd.read_csv("pnas_table_mod.csv")
bound = list(data["bound"])
closed = list(data["closed"])
directories = data["af_pdb"]
ligs = list(data["lig_id"])
residues = data["segment"].apply(parse_entry).to_list()

dirs = [x.split(".")[0] for x in directories]
bound_pdbs = [x.replace(".", "_") for x in bound]
closed_pdbs = [x.replace(".", "_") for x in closed]

pdb_directory = Path("pdb_structures_final/")
af3_lig_directory = Path("pnas_af3_lig")
af3_nolig_directory = Path("pnas_af3_nolig")

for i, pdb_dir in enumerate(pdb_directory.iterdir()):
    if not pdb_dir.name.startswith("1RHB"):
        continue

    counts_dict = {"State": [], "Count": []}
    clash_dict = {"model": [], "clash": []}

    protein = pdb_dir.name.split("_")[0]
    prot_index = dirs.index(protein)
    bound_pdb = bound_pdbs[prot_index]
    closed_pdb = closed_pdbs[prot_index]
    ref_lig_struct = parser.get_structure(
        "ref", pdb_directory / f"{protein.split('_')[0]}_pdbs" / f"{bound_pdb}.cif"
    )
    ref_chain = ref_lig_struct[0][bound_pdb.split("_")[-1]]
    for state in ["bound", "unbound"]:
        clashing = 0
        total = 0
        for file in Path(pdb_dir / state).glob("*.cif"):
            clash_dict["model"].append(file.stem)
            try:
                test_struct = parser.get_structure("test", file)
                test_chain = test_struct[0][file.stem.split("_")[-1]]
                residue_map = map_residues_by_coordinates(ref_chain, test_chain)
                residues_mapped = [
                    residue_map[x] for x in residues[prot_index] if x != 218
                ]
                clashes = count_clashes(test_struct, ref_lig_struct, residues_mapped)
                if clashes:
                    clash_dict["clash"].append("yes")
                    clashing += 1
                else:
                    clash_dict["clash"].append("no")
                total += 1
            except KeyError:
                try:
                    print(file)
                    alt_lig_struct = parser.get_structure(
                        "ref",
                        pdb_directory
                        / f"{protein.split('_')[0]}_pdbs"
                        / f"{closed_pdb}.cif",
                    )
                    alt_chain = alt_lig_struct[0][closed_pdb.split("_")[-1]]
                    residue_map = map_residues_by_coordinates(alt_chain, test_chain)
                    residues_mapped = [
                        residue_map[x] for x in residues[prot_index] if x != 218
                    ]
                    clashes = count_clashes(
                        test_struct, ref_lig_struct, residues_mapped
                    )
                    if clashes:
                        clash_dict["clash"].append("yes")
                        clashing += 1
                    else:
                        clash_dict["clash"].append("no")
                    total += 1
                except KeyError:
                    clash_dict["clash"].append("error")
                    print(f"Error in: {file.name}")
                    continue
        counts_dict["State"].append(f"{protein}_{state}_clashing")
        counts_dict["Count"].append(clashing)
        counts_dict["State"].append(f"{protein}_{state}_non_clashing")
        counts_dict["Count"].append(total - clashing)
        df = pl.DataFrame(clash_dict)
        df.write_csv(f"count_by_clash/{pdb_dir.stem}_PDB_{state}.csv")

print("PDBs:")
print(counts_dict)


for i, pdb_dir in enumerate(af3_nolig_directory.iterdir()):
    if not pdb_dir.name.startswith("1rhb"):
        continue

    counts_dict = {"State": [], "Count": []}
    clash_dict = {"model": [], "clash": []}

    protein = pdb_dir.name.upper()
    prot_index = dirs.index(protein)
    bound_pdb = bound_pdbs[prot_index]
    print(bound_pdb)
    ref_lig_struct = parser.get_structure(
        "ref",
        pdb_directory
        / f"{protein.split('_')[0]}_pdbs"
        / f"bound/aligned_to_open/{bound_pdb}.cif",
    )
    ref_chain = ref_lig_struct[0][bound_pdb.split("_")[-1]]
    clashing = 0
    total = 0
    for file in Path(af3_nolig_directory, protein.lower(), "aligned_to_open").glob(
        "*.cif"
    ):
        test_struct = parser.get_structure("test", file)
        test_chain = test_struct[0]["A"]
        residue_map = map_residues_by_coordinates(ref_chain, test_chain)
        residues_mapped = [residue_map[x] for x in residues[prot_index] if x != 218]
        clashes = count_clashes(test_struct, ref_lig_struct, residues_mapped)
        clash_dict["model"].append(file.stem)
        if clashes:
            clash_dict["clash"].append("yes")
            clashing += 1
        else:
            clash_dict["clash"].append("no")
        total += 1
    counts_dict["State"].append(f"{protein}_clashing")
    counts_dict["Count"].append(clashing)
    counts_dict["State"].append(f"{protein}_non_clashing")
    counts_dict["Count"].append(total - clashing)
    df = pl.DataFrame(clash_dict)
    df.write_csv(f"count_by_clash/{pdb_dir.stem}_af3_nolig.csv")

    print("AF3 nolig:")
    print(counts_dict)

for i, pdb_dir in enumerate(af3_lig_directory.iterdir()):
    if not pdb_dir.name.startswith("1rhb"):
        continue

    counts_dict = {"State": [], "Count": []}
    clash_dict = {"model": [], "clash": []}

    protein = pdb_dir.name.upper()
    prot_index = dirs.index(protein)
    bound_pdb = bound_pdbs[prot_index]
    print(bound_pdb)
    ref_lig_struct = parser.get_structure(
        "ref",
        pdb_directory
        / f"{protein.split('_')[0]}_pdbs"
        / f"bound/aligned_to_open/{bound_pdb}.cif",
    )
    ref_chain = ref_lig_struct[0][bound_pdb.split("_")[-1]]
    clashing = 0
    total = 0
    for file in Path(af3_lig_directory, protein.lower(), "aligned_to_open").glob(
        "*.cif"
    ):
        test_struct = parser.get_structure("test", file)
        test_chain = test_struct[0]["A"]
        residue_map = map_residues_by_coordinates(ref_chain, test_chain)
        residues_mapped = [residue_map[x] for x in residues[prot_index] if x != 218]
        clashes = count_clashes(test_struct, ref_lig_struct, residues_mapped)
        clash_dict["model"].append(file.stem)
        if clashes:
            clash_dict["clash"].append("yes")
            clashing += 1
        else:
            clash_dict["clash"].append("no")
        total += 1
    counts_dict["State"].append(f"{protein}_clashing")
    counts_dict["Count"].append(clashing)
    counts_dict["State"].append(f"{protein}_non_clashing")
    counts_dict["Count"].append(total - clashing)
    df = pl.DataFrame(clash_dict)
    df.write_csv(f"count_by_clash/{pdb_dir.stem}_af3_lig.csv")

    print("AF3 lig:")
    print(counts_dict)
