"""Mamba Generator"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class Generator(nn.Module):
    """Generator function: Generate time-series data in latent space using Mamba."""
    def __init__(self, config):
        super(Generator, self).__init__()

        self.hidden_dim = config['hidden_dim']  # 256
        self.seq_len = config['seq_len']  # 150

        self.l1 = nn.Linear(config.get('z_dim', 100), self.seq_len * self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.hidden_dim))

        self.mamba_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                Mamba(
                    d_model=self.hidden_dim,            # Input dim (features)
                    d_state=config.get('d_state', 2),   # Hidden state (latent dim)
                    d_conv=config.get('d_conv', 2),     # Temporal abstraction
                    expand=config.get('e_fact', 1),     # Expansion factor
                )
            ) for _ in range(config.get('num_layers', 1))
        ])

        self.fc = nn.Linear(self.hidden_dim, config['channels'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_noise, sigmoid=True):
        """Forward func"""
        x = self.l1(input_noise).view(-1, self.seq_len, self.hidden_dim)

        mamba_input = x + self.pos_embed

        for mamba in self.mamba_layers:
            mamba_input = mamba(mamba_input)  # shape: (batch, seq_len, z_dim)

        e = self.fc(mamba_input)  # shape: (batch, seq_len, feat_dim)
        if sigmoid:
            e = self.sigmoid(e)
        return e
