"""Script to add ligand pLDDT values to the pose RMSD csv files
Author: Maria L.
"""

from pathlib import Path
import polars as pl
from Bio.PDB.MMCIFParser import MMCIFParser
import numpy as np
from tqdm import tqdm

parser = MMCIFParser(QUIET=True)

rmsd_csvs = Path("pose_rmsds").glob("*_RMSDs.csv")
af_models = Path("pnas_af3_lig")
for file in tqdm(rmsd_csvs):
    pLDDT = []
    df = pl.read_csv(file)
    for model in df["pdb_seed"]:
        pdb, seed = model[:4], model[5:]
        cif_path = af_models / f"{pdb}" / f"{seed}" / "model.cif"
        structure = parser.get_structure(f"{model}", str(cif_path))
        het_bfacts = [
            atom.bfactor
            for chain in structure.get_chains()
            for residue in chain
            if residue.id[0].startswith("H_")
            for atom in residue
        ]
        pLDDT.append(np.mean(het_bfacts))

    plddt_df = pl.DataFrame({"pLDDT": pLDDT})
    out_df = df.hstack(plddt_df)
    out_df.write_csv(file.with_name(file.stem + "_with_pLDDT.csv"))
