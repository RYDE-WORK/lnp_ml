import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import List
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import (
    Mol,
    AllChem,
    MACCSkeys,
    Descriptors
)

mp.set_start_method('fork')  # Screw MacOS
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_morgan(mol: Mol, radius: int = 2, nBits: int = 1024) -> List[int]:
    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=nBits,
        useChirality=False
    ).ToList()


def get_maccs(mol: Mol) -> List[int]:
    return MACCSkeys.GenMACCSKeys(mol).ToList()


def get_rdkit_descriptors(mol: Mol) -> List[float]:
    desc_dict = Descriptors.CalcMolDescriptors(mol)
    return list(desc_dict.values())