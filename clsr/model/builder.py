from clsr.model.autoencoder import AutoEncoder
from clsr.model.classifier import Classifier

models_dict = {
    'autoencoder': AutoEncoder,
    'classifier': Classifier,
}


def build_model(name: str, *args, **kwargs):
    return models_dict[name](*args, **kwargs)