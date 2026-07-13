import torch.nn as nn


class ReCoBranchDPT(nn.Module):
    def __init__(self, in_channels, output_dim=256):
        super().__init__()
        self.representation = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 1),
        )

    def forward(self, features):
        return self.representation(features)
