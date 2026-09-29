"""CoverTypeClassifier — identical architecture to the notebook.

Layer indices inside ``self.layers`` are preserved, so a ``state_dict`` saved
directly from the notebook (``torch.save(model.state_dict(), ...)``) loads here too.
"""

from __future__ import annotations

import torch
from torch import nn


class CoverTypeClassifier(nn.Module):
    """MLP: 54 -> 128 -> 128 -> 128 -> 64 -> 32 -> 7 with BatchNorm + ReLU.

    Note: the output layer is followed by ``BatchNorm1d(num_class)`` on the logits,
    exactly as in the notebook.
    """

    def __init__(self, input_size: int, num_class: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, num_class), nn.BatchNorm1d(num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
