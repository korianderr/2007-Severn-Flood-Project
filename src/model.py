"""
model.py

A small U-Net (2 encoder/decoder levels, filters 8-16-32)
for the flood-depth emulator. Kept small given only 18 training
scenarios and CPU-only training - the standard U-Net paper's sizing
(64+ filters, 4-5 levels) assumes far more training data than this
project has, and would be far more likely to overfit or be
impractically slow here.
"""

import torch
import torch.nn as nn

class FloodUNet(nn.Module):
    """
    Predicts a flood depth grid from a 2-channel input (elevation +
    broadcast boundary_wse), same spatial size in and out.

    Encoder: 2 -> 8 -> 16 -> 32 (bottleneck), halving spatial size at
    each pooling step. Decoder mirrors this back up to full resolution,
    concatenating each level's skip connection along the channel axis
    to recover spatial detail lost during pooling.
    """

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.up1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        self.out = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: input tensor, shape (batch, 2, H, W).

        Returns:
            predicted depth tensor, shape (batch, 1, H, W).
        """
        e1 = torch.relu(self.enc1(x))
        p1 = self.pool1(e1)

        e2 = torch.relu(self.enc2(p1))
        p2 = self.pool2(e2)

        b = torch.relu(self.bottleneck(p2))

        u2 = self.up2(b)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = torch.relu(self.dec2(d2))

        u1 = self.up1(d2)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = torch.relu(self.dec1(d1))

        return self.out(d1)
