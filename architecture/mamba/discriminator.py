import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np
from mamba_ssm import Mamba

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class Discriminator(nn.Module):
    """Generator function: Generate time-series data in latent space using Mamba."""
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.l1 = nn.Linear(config['channels'], config.get('hidden_dim', 256))

        self.pos_embed = nn.Parameter(torch.zeros(1, config['seq_len'], config.get('hidden_dim', 256)))

        self.mamba_layers = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(config.get('hidden_dim', 256)),
                    Mamba(
                        d_model=config.get('hidden_dim', 256),      # Input dimension: latent_dim from Encoder/Generator
                        d_state=config.get('d_state', 256),         # Mamba's internal hidden state dimension
                        d_conv=config.get('d_conv', 2),             # Convolutional depth
                        expand=config.get('e_fact', 1),             # Expansion factor
                    )
                ) for _ in range(config.get('num_layers', 1))
            ])

        self.fc_classify = nn.Linear( config['seq_len']* config.get('hidden_dim', 256) , 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq, sigmoid=True):
        x = self.l1(input_seq)

        mamba_input = x + self.pos_embed

        for mamba in self.mamba_layers:
            mamba_input = mamba(mamba_input) 
        
        E = self.fc_classify(mamba_input.view(mamba_input.size(0), -1)) 
        if sigmoid:
            E = self.sigmoid(E)
        return E
