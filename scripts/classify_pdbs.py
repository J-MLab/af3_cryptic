"""This script classifies PDB structures into 'bound' and 'unbound' categories based on
reference cryptic site location and clashing residues information.
This version of the script skips only forms of water as ligands
Authors: Maria L. and Felix T.
Date: June 2025"""

import os
import pymol
from pymol import cmd
from Bio.PDB import MMCIFParser, is_aa
import numpy as np
import pandas as pd
import shutil
from scripts.find_clashes_in_different_structs import count_clashes
from Bio import PDB

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
refs = [
    x if x != data["bound"][i] else data["open"][i]
    for i, x in enumerate(data["closed"])
]
bound = list(data["bound"])
directories = data["af_pdb"]
ligs = list(data["lig_id"])
residues = data["segment"].apply(parse_entry).to_list()

dirs = [x.replace(".", "_") for x in directories]
ref_pdbs = [x.replace(".", "_") for x in refs]
bound_pdbs = [x.replace(".", "_") for x in bound]

pdb_directory = "pdb_structures_final/"

for i, protein in enumerate(dirs):
    if protein not in ['1RHB_A']:
        continue
    print(f"Processing files for {protein} and ligand {ligs[i]}")
    reference_file = [
        file
        for file in os.listdir(f"{pdb_directory}/{protein[:-2]}_pdbs/")
        if file.startswith(f"{bound_pdbs[i]}")
    ]
    reference_structure = f"{pdb_directory}/{protein[:-2]}_pdbs/{reference_file[0]}"
    unbound_structure = f"{pdb_directory}/{protein[:-2]}_pdbs/{ref_pdbs[i]}.cif"
    ref_struct = parser.get_structure(protein, reference_structure)
    unbound_struct = parser.get_structure(protein, unbound_structure)
    ligand_residue_name = ligs[i]
    ref_center = get_ligand_instances(reference_structure, ligand_residue_name)
    cmd.load(reference_structure, "ref")
    # iterate through the pdb folders and the .cif's inside of them
    for file in os.listdir(f"{pdb_directory}/{protein[:-2]}_pdbs/"):  # usually adjac
        if file != f"{bound_pdbs[i]}.cif" and file.endswith(".cif"):
            test_structure = f"{pdb_directory}/{protein[:-2]}_pdbs/{file}"
            test_struct = parser.get_structure(file[:-4], test_structure)
            # create bound and unbound directories
            os.makedirs(f"{pdb_directory}/{protein[:-2]}_pdbs/bound", exist_ok=True)
            os.makedirs(f"{pdb_directory}/{protein[:-2]}_pdbs/unbound", exist_ok=True)
            print(count_clashes(
                unbound_struct, test_struct, residues
            ))
            if check_ligand_binding(test_structure, ref_center) and count_clashes(
                unbound_struct, test_struct, residues[i]
            ):    # if at same site --> add to bound folder
                shutil.copy(
                    f"{pdb_directory}/{protein[:-2]}_pdbs/{file}",
                    f"{pdb_directory}/{protein[:-2]}_pdbs/bound/{file}",
                )
            else:  # add to unbound if not
                shutil.copy(
                    f"{pdb_directory}/{protein[:-2]}_pdbs/{file}",
                    f"{pdb_directory}/{protein[:-2]}_pdbs/unbound/{file}",
                )
            cmd.delete("test")
    cmd.delete("ref")
    shutil.copy(
        f"{pdb_directory}/{protein[:-2]}_pdbs/{bound_pdbs[i]}.cif",
        f"{pdb_directory}/{protein[:-2]}_pdbs/bound/{bound_pdbs[i]}.cif",
    )
