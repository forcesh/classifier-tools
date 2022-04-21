import torch.nn as nn

from clsr.model.encoder import Encoder
from clsr.model.decoder import Decoder


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, input_dim)

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        return x