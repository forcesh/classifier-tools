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
                 freeze_encoder=True,
                 **kwargs):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        if freeze_encoder:
            for n, p in self.encoder.named_parameters():
                p.requires_grad = False
        self.cls_head = nn.Linear(in_features=hidden_dim,
                                  out_features=num_classes)

        if encoder_weights is not None and os.path.exists(encoder_weights):
            self._load_encoder_weigths(encoder_weights)
            print(encoder_weights, 'encoder weights loaded')

    def _load_encoder_weigths(self, encoder_weights):
        checkpoint = torch.load(encoder_weights)
        new_checkpoint = {}
        for k, v in checkpoint.items():
            if 'decoder.' in k:
                continue
            new_k = k.replace('encoder.', '')
            new_checkpoint[new_k] = v
        self.encoder.load_state_dict(new_checkpoint)

    def forward(self, input):
        x = input.view(input.size(0), -1)
        x = self.encoder(x)
        x = self.cls_head(x)
        return x
