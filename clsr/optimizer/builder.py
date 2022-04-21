import torch.optim as optim

optimizers_dict = {
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
}


def build_optimizer(name, *args, **kwargs):
    return optimizers_dict[name](*args, **kwargs)
