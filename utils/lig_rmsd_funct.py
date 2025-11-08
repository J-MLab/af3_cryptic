from Bio.PDB import MMCIFParser, Superimposer, PDBIO, PDBParser
import numpy as np
import os
import pandas as pd
from rdkit import Chem
import pymol2

def strip_stereochemistry(mol):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    for bond in mol.GetBonds():
        bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
    return mol

def structure_to_loose_smarts(mol):
    mol = Chem.RWMol(Chem.Mol(mol))
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.UNSPECIFIED)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return Chem.MolToSmarts(mol, isomericSmiles=False)

def parse_cif_and_align(ref_cif, pred_cif):
    os.makedirs('prot_temp', exist_ok=True)
    ref_id = "ref"
    pred_id = "pred"
    with pymol2.PyMOL() as pymol:
        pymol.cmd.load(ref_cif, ref_id)
        pymol.cmd.load(pred_cif, pred_id)
        pymol.cmd.align(
            f"{pred_id} and name N+CA+C and polymer.protein",
            f"{ref_id} and name N+CA+C and polymer.protein"
        )
        aligned_pdb = "prot_temp/s2_aligned.pdb"
        pymol.cmd.save(aligned_pdb, pred_id)
    pparser = PDBParser(QUIET=True)
    cparser = MMCIFParser(QUIET=True)
    s1 = cparser.get_structure('s1', ref_cif)
    s2 = pparser.get_structure('s2', aligned_pdb)
    return s1, s2

df = pd.read_csv("pnas_table_mod.csv")
def get_ligand_id_by_bound_prefix(bound_prefix, table):
    mask = table['bound'].str[:4] == bound_prefix[:4]
    matches = table.loc[mask, ['bound', 'lig_id']]
    if matches.empty:
        print(f"No entries found with bound starting '{bound_prefix}'")
        return None
    if len(matches) > 1:
        print(f"Multiple matches for prefix '{bound_prefix}':")
        print(matches.to_string(index=False))
        return matches['lig_id'].tolist()
    return matches.iloc[0]['lig_id']

def get_molecule_struct(ref_struct, pred_struct, lig_ID):
    os.makedirs('ligand_temp', exist_ok=True)

    # Extract ref ligand
    ref_ligand = None
    for model in ref_struct:
        for chain in model:
            for residue in chain:
                if residue.id[0].startswith('H_') and residue.resname == lig_ID:
                    ref_ligand = residue
                    break
    if ref_ligand is None:
        print(f"[DEBUG] No reference ligand found matching {lig_ID} in {ref_struct.id}")
    else:
        print(f"[DEBUG] Found reference ligand: {ref_ligand.resname} at {ref_ligand.id}")

    io = PDBIO()
    io.set_structure(ref_ligand)
    io.save("ligand_temp/ref.pdb")

    # Extract predicted ligand (first non-water HETATM)
    pred_ligand = None
    for model in pred_struct:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                hetero_flag = residue.id[0]
                if residue.id[0].startswith('H_') and residue.resname != 'HOH' and chain_id != 'C':
                    print(residue.resname)
                    pred_ligand = residue
                    break
    if pred_ligand is None:
        print(f"[DEBUG] No predicted ligand found in {pred_struct.id}")
    else:
        print(f"[DEBUG] Found predicted ligand: {pred_ligand.resname} at {pred_ligand.id}")

    io.set_structure(pred_ligand)
    io.save("ligand_temp/pred.pdb")

    ref_mol = Chem.MolFromPDBFile("ligand_temp/ref.pdb", removeHs=False, sanitize=True)
    pred_mol = Chem.MolFromPDBFile("ligand_temp/pred.pdb", removeHs=False, sanitize=True)

    ref_mol = Chem.RemoveHs(ref_mol)
    pred_mol = Chem.RemoveHs(pred_mol)

    if ref_mol is None:
        print("[ERROR] RDKit failed to read ligand_temp/ref.pdb as a molecule")
    else:
        print(f"[DEBUG] RDKit molecule constructed for ref: {Chem.MolToSmiles(ref_mol)}")
    if pred_mol is None:
        print("[ERROR] RDKit failed to read ligand_temp/pred.pdb as a molecule")
    else:
        print(f"[DEBUG] RDKit molecule constructed for pred: {Chem.MolToSmiles(pred_mol)}")

    Chem.SanitizeMol(ref_mol)
    ref_mol = strip_stereochemistry(ref_mol)
    Chem.SanitizeMol(pred_mol)
    pred_mol = strip_stereochemistry(pred_mol)

    ref_coords = get_mol_coords(ref_mol)
    pred_coords = get_mol_coords(pred_mol)
    cs_smarts = structure_to_loose_smarts(ref_mol)
    mappings = cs_sym_mappings(ref_mol, pred_mol, cs_smarts)

    return ref_coords, pred_coords, mappings

def cs_sym_mappings(ref_mol, pred_mol, cs_smarts):
    patt = Chem.MolFromSmarts(cs_smarts)
    ref_matches = ref_mol.GetSubstructMatches(patt, uniquify=False)
    pred_matches = pred_mol.GetSubstructMatches(patt, uniquify=False)

    mappings = []
    for ref_idx in ref_matches:
        for pred_idx in pred_matches:
            mapping = list(zip(ref_idx, pred_idx))  # (ref_atom, pred_atom)
            mappings.append(mapping)
    if not mappings:
        print("[DEBUG] No valid atom mappings found")
    return mappings

def get_mol_coords(mol):
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append((pos.x, pos.y, pos.z))
    return coords

def calc_rmsd_from_mapping(ref_coords, pred_coords, mapping):
    n_ats = len(mapping)
    mapped_ref_coords = np.zeros((n_ats, 3))
    mapped_pred_coords = np.zeros((n_ats, 3))
    for i, (ref_at, pred_at) in enumerate(mapping):
        mapped_ref_coords[i][:] = ref_coords[ref_at]
        mapped_pred_coords[i][:] = pred_coords[pred_at]
    rmsd = np.sqrt(np.mean(np.sum((mapped_ref_coords - mapped_pred_coords) ** 2, axis=1)))
    return rmsd

def calculate_pose_rmsd(ref_cif, pred_cif):
    ref_struct, pred_struct = parse_cif_and_align(ref_cif, pred_cif)
    lig_ID = get_ligand_id_by_bound_prefix(ref_cif.split('/')[-1][:-4], df)
    ref_coord, pred_coord, mappings = get_molecule_struct(ref_struct, pred_struct, lig_ID)

    if not mappings:
        raise ValueError("❌ No valid atom mappings found")

    best_rmsd = float('inf')
    best_idx = -1
    print(f"Found {len(mappings)} symmetry-correct atom mappings. Calculating RMSDs:")
    for idx, mapping in enumerate(mappings):
        rmsd = calc_rmsd_from_mapping(ref_coord, pred_coord, mapping)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_idx = idx
            print(f"    ↳ New best RMSD so far!")

    print(f"Best RMSD is mapping {best_idx+1} with RMSD = {best_rmsd:.3f} Å")
    return best_rmsd