import torch
import torch.nn as nn
from typing import Dict


class TabularEncoder(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tabular_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Input: Dict with keys 'comp', 'phys', 'help', 'exp'
        #   Each value is a tensor [B, D_i] where D_i is the feature dimension
        # Output: Same dict (pass-through, features already grouped by DataLoader)
        
        # The DataLoader (trainer.py) already groups features correctly:
        # - 'comp': [B, 9] - composition features
        # - 'phys': [B, 9] - physical features (including processed PDI)
        # - 'help': [B, 4] - helper lipid one-hot features
        # - 'exp': [B, 20] - experimental condition one-hot features (including processed Purity)
        
        # Simply return the dict as-is
        # If we wanted to add learned transformations, we could add linear layers here
        return tabular_data
