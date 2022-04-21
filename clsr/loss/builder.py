from torch import nn


losses_dict = {
    'mse': nn.MSELoss,
    'ce': nn.CrossEntropyLoss,
}


def build_loss(name: str, *args, **kwargs) -> nn.Module:
    return losses_dict[name](*args, **kwargs)
