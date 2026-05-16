"""Featurization functions for chemical data. This module provides utilities to convert PSMILES strings into Morgan fingerprints as numpy arrays."""
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


## Utility functions for featurization

def psmiles_to_mol(psmile: str) -> Chem.Mol:
    """Convert a PSMILES string to an RDKit molecule."""
    return Chem.MolFromSmiles(psmile)


def mol_to_morganfingerprint(mol: Chem.Mol, radius: int = 2) -> rdkit.DataStructs.cDataStructs.ExplicitBitVect:
    """Convert an RDKit molecule to a Morgan fingerprint. Use 2048 bits by default."""
    fpgen = AllChem.GetMorganGenerator(radius=radius)
    return fpgen.GetFingerprint(mol)


def morganfingerprints_to_nparray(fps: list[rdkit.DataStructs.cDataStructs.ExplicitBitVect]) -> np.ndarray:
    """
    Convert a list of Morgan fingerprints to a numpy array.
    The resulting array will have shape (num_samples, num_bits), where num_bits is the length of the fingerprint (e.g., 2048).
    Each row corresponds to a molecule, and each column corresponds to a bit in the fingerprint.
    """
    fp_arrays = []
    for fp in fps:
        fp_array = np.zeros((fp.GetNumBits(),), dtype=np.int32)
        for bit in fp.GetOnBits():
            fp_array[bit] = 1
        fp_arrays.append(fp_array)
    return np.array(fp_arrays)


## Wrappers for the above functions to process lists of PSMILES strings

def psmiles_to_morganfingerprints(psmiles: list[str], radius: int = 2) -> list[rdkit.DataStructs.cDataStructs.ExplicitBitVect]:
    """Convert PSMILES list to a list of Morgan fingerprints."""
    fps = []
    for psmile in psmiles:
        mol = psmiles_to_mol(psmile)
        if mol is None:
            raise ValueError(f"Could not parse PSMILES '{psmile}' into a molecule.")
            
        fp = mol_to_morganfingerprint(mol, radius=radius)
        if fp is None:
            raise ValueError(f"Could not generate fingerprint for PSMILES '{psmile}'.")
        fps.append(fp)
    return fps


def psmiles_to_nparray(psmiles: list[str], radius: int = 2) -> np.ndarray:
    """Convert a list of PSMILES strings to a numpy array of Morgan fingerprints."""
    fps = psmiles_to_morganfingerprints(psmiles, radius=radius)
    return morganfingerprints_to_nparray(fps)


psmiles = ["[*]c1ccc(Oc2ccc(Sc3ccc(Oc4ccc(-c5nc([*])nc(-c6ccccc6)n5)cc4)cc3)cc2)cc1", "[*]CC([*])(C)C(=O)OC1CCOCC1"]
if __name__ == "__main__":
#     # for data_path in DATA_PATHS:
#     #     print(f"Processing {data_path}")
#     #     psmiles_list = load_psmiles(data_path)
#     #     test_rdkit_parsing(psmiles_list)
#     #     test_morgan_fingerprints(psmiles_list)
#     # print(f"Processing {VALIDATION_PATH}")
#     # psmiles_list = load_psmiles(VALIDATION_PATH)
    fps = psmiles_to_morganfingerprints(psmiles)
    np_array = morganfingerprints_to_nparray(fps)
    print(np_array.shape)
    print(type(fps[0]))
    print(fps[0].GetNumBits())
    print(fps[0].GetNumOnBits())
    print(list(fps[0].GetOnBits()))
