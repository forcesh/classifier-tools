import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_dim, outpud_dim, **kwargs):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=outpud_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)