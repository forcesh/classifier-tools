import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU()
        )

    def forward(self, input):
        return self.model(input)