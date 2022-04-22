import torch.nn as nn

from clsr.model.decoder import Decoder
from clsr.model.encoder import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        print('hidden_dim =', hidden_dim)
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, input_dim)

    def forward(self, input):
        x = input.view(input.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(input.size())
        return x
