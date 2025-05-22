import torch

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"  # 20 amino acids
AA_TO_INDEX = {aa: idx + 1 for idx, aa in enumerate(AA_LIST)}  # Start from 1
AA_TO_INDEX['X'] = 0  # Unknown AA or padding

def encode_sequence(seq: str):
    return torch.tensor([AA_TO_INDEX.get(aa, 0) for aa in seq], dtype=torch.long)

