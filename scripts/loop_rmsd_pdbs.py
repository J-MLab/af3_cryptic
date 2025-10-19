import os
import numpy as np
import csv
import argparse
from Bio.PDB import PDBParser, NeighborSearch, Superimposer, is_aa, PDBIO
from Bio.PDB.Polypeptide import PPBuilder
from pymol import cmd
import pandas as pd
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1

"""Script to calculate RMSD between cryptic site residues in aligned structures and their 
crystal structure reference. Moving segments are defined in pnas_table.csv. Only for continuous 
moving segments.
Author: Maria L.
"""


def download_pdb(pdb_id):
    """
    Downloads a PDB file using PyMOL's fetch command and saves it locally.
    Args:
        pdb_id (str): The PDB ID of the structure to download.
    Returns:
        str: Path to the downloaded PDB file.
    """
    try:
        # Extract only the PDB ID (strip chain information if present)
        pdb_id_clean = pdb_id.split(".")[0].lower()

        # Fetch the structure
        cmd.fetch(pdb_id)

        # Save the structure as a CIF file
        output_file = f"{pdb_id_clean}.cif"
        cmd.save(output_file, pdb_id)

        # Delete the fetched object from PyMOL
        cmd.delete("*")

        return output_file
    except Exception as e:
        print(f"Error downloading PDB {pdb_id}: {e}")
        return None


def safe_three_to_one(resname):
    try:
        return protein_letters_3to1[resname.capitalize()]
    except KeyError:
        return None


def load_structure(pdb_file, structure_id):
    """
    Loads a CIF structure from a file.
    Args:
        pdb_file (str): Path to the CIF file.
        structure_id (str): An identifier for the structure.
    Returns:
        Structure: The loaded structure.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, pdb_file)
    return structure


from scipy.spatial.distance import cdist


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
        # Filter mobile residues of the same type
        matching_indices = [
            j for j, mob_type in enumerate(mobile_types) if mob_type == ref_type
        ]

        if matching_indices:
            # Extract distances to matching residues
            matching_distances = distances[i, matching_indices]
            min_idx_in_matches = matching_distances.argmin()
            min_idx = matching_indices[min_idx_in_matches]

            # Check if the minimum distance is within the threshold
            if distances[i, min_idx] <= threshold:
                residue_map[ref_res] = mobile_residues_list[min_idx]

    return residue_map


def find_interacting_residues(structure, startres, endres):
    """
    Finds residues in the structure that interact with the ligand residues without using NeighborSearch.
    Args:
        structure: The structure to search.
        ligand_residues (list): List of ligand residues.
        distance_cutoff (float): Distance cutoff in Å for interaction.
    Returns:
        set: A set of interacting residues.
    """
    interacting_residues = set()

    # Loop over all residues in the structure
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue):
                    continue  # Skip non-amino acid residues
                elif residue.id[1] >= startres and residue.id[1] <= endres:
                    interacting_residues.add(residue)

    return interacting_residues


def calculate_rmsd(
    ref_structure, ref_chain_id, mobile_structure, mobile_chain_id, interacting_residues
):
    """
    Calculates RMSD between reference and mobile structures using only alpha-carbon atoms.
    Args:
        ref_structure: Reference structure.
        ref_chain_id: Chain ID for the reference structure.
        mobile_structure: Mobile structure to be aligned.
        mobile_chain_id: Chain ID for the mobile structure.
        interacting_residues: List of residues to include in RMSD calculation.
    Returns:
        float: RMSD value, or None if no matching atoms are found.
    """
    ref_chain = ref_structure[0][ref_chain_id]
    mobile_chain = mobile_structure[0][mobile_chain_id]
    # Map residues by Cα atom proximity
    residue_map = map_residues_by_coordinates(ref_chain, mobile_chain)
    ref_atoms = []
    mobile_atoms = []
    for ref_res in interacting_residues:
        if ref_res in residue_map:
            mobile_res = residue_map[ref_res]
            # Only add alpha-carbon atoms (CA)
            if "CA" in ref_res and "CA" in mobile_res:
                ref_atoms.append(ref_res["CA"])
                mobile_atoms.append(mobile_res["CA"])

    # Check if we have enough atoms to calculate RMSD
    if len(ref_atoms) == 0:
        print("No matching alpha-carbon atoms found for RMSD calculation.")
        return None
    return calculate_static_rmsd(ref_atoms, mobile_atoms)


def calculate_static_rmsd(ref_atoms, mobile_atoms):
    """
    Calculate static RMSD between two sets of atoms.
    Args:
        ref_atoms (list): List of reference atom objects.
        mobile_atoms (list): List of mobile atom objects.
    Returns:
        float: RMSD value.
    """
    if len(ref_atoms) != len(mobile_atoms):
        raise ValueError("The number of atoms in the two structures must be the same.")

    # Extract coordinates
    ref_coords = np.array([atom.get_coord() for atom in ref_atoms])
    mobile_coords = np.array([atom.get_coord() for atom in mobile_atoms])

    # Calculate the squared differences
    diff = ref_coords - mobile_coords
    squared_diff = np.sum(diff**2, axis=1)

    # Compute RMSD
    rmsd = np.sqrt(np.mean(squared_diff))
    return rmsd


def main(path, startres, stopres, refchain, refpdbs):
    # For each subdirectory in the given path
    bu_directories = ["unbound", "bound"]
    directories = ["aligned_to_open", "aligned_to_closed"]

    for budi in bu_directories:
        for j, dir in enumerate(directories):
            pocket_rmsd = {"model": [], "loop_rmsd": []}
            dirpath = os.path.join(path, budi, dir)
            for file in os.listdir(dirpath):
                pdb_path = os.path.join(dirpath, file)

                # Load the aligned structure
                aligned_structure = load_structure(pdb_path, "aligned")

                # Download and load the crystal structure
                crystal_pdb_path = download_pdb(refpdbs[j])
                crystal_structure = load_structure(crystal_pdb_path, "crystal")

                # Find residues interacting with the ligand
                interacting_residues = find_interacting_residues(
                    crystal_structure, startres, stopres
                )
                print(f"Number of interacting residues: {len(interacting_residues)}")

                if not interacting_residues:
                    print("No interacting residues found.")
                    continue

                # Superimpose the structures using backbone atoms
                # Calculate RMSD between interacting residues
                print(file)
                rmsd = calculate_rmsd(
                    crystal_structure,
                    refchain[j],
                    aligned_structure,
                    file.split("_")[-1].strip(".cif"),
                    interacting_residues,
                )
                pocket_rmsd["model"].append(file[:-4])
                pocket_rmsd["loop_rmsd"].append(rmsd)

            df = pd.DataFrame(pocket_rmsd)
            dirn = dir.split("_")[-1]
            df.to_csv(f"{path}/{budi}/{dirn}_rmsd.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate RMSD between interacting residues in aligned structures and their crystal structures."
    )
    parser.add_argument("path", type=str, help="Path to one of 16 directories (full)")
    args = parser.parse_args()
    csv_file = "pnas_table.csv"
    subdir = args.path.split("/")[-1]

    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    print(subdir)
    matching_entry = next(
        (
            entry
            for entry in data
            if entry["af_pdb"].startswith(subdir.strip("_pdbs").upper())
        ),
        None,
    )
    residues = matching_entry["segment"].split(" ")
    startres = int(residues[0][3:])
    endres = int(residues[2][3:])
    refpdbs = [matching_entry["open"], matching_entry["closed"]]
    refchains = [
        matching_entry["open"].split(".")[-1],
        matching_entry["closed"].split(".")[-1],
    ]
    print(args.path, startres, endres, refchains, refpdbs)
    main(args.path, startres, endres, refchains, refpdbs)
