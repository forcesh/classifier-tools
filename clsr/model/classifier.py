import os

import torch
import torch.nn as nn

from clsr.model.encoder import Encoder


class Classifier(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_classes,
                 encoder_weights=None,
                 **kwargs):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        if encoder_weights is not None and os.path.exists(encoder_weights):
            checkpoint = torch.load(encoder_weights)
            # ToDo load only encoder weights
            self.encoder.load_state_dict(checkpoint)
            print(encoder_weights, ' encoder weights loaded')
        self.cls_head = nn.Linear(in_features=hidden_dim,
                                  out_features=num_classes)

    def forward(self, input):
        x = input.view(input.size(0), -1)
        x = self.encoder(input)
        x = self.cls_head(x)
        return x
