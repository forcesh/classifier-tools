# classifier-tools

## Quick start
Install python>=3.9 and cuda11
```
cd classifier-tools
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
```
python setup.py install
```
or
```
pip install -e .
```

## Development
Install packages for code style, pip-tools
```
pip install -r requirements.dev.txt -f https://download.pytorch.org/whl/torch_stable.html
pre-commit install
```

### Pip compile
```
pip-compile -r requirements.in -f https://download.pytorch.org/whl/torch_stable.html --verbose
```
for new packages type
```
pip-compile --upgrade-package <package_name> --verbose
```
### Download dataset
```
bash tools/mnist_get_data.sh
```
### Training, testing and generating masks
```
python tools/train.py --config-name="autoencoder.yaml"
python tools/train.py --config-name="classifier.yaml"
python tools/test.py --config-name="autoencoder.yaml"
python tools/test.py --config-name="classifier.yaml"
```

### Results:
| Model                                          | F1 score    |
|------------------------------------------------|-------------|
| hidden_dim: 128 (pretrain; no freezed encoder) | 0.9688 |
| hidden_dim: 128 (pretrain; freezed encoder)    | 0.8361 |
| **hidden_dim: 128 (no pretrain)**                  | **0.9738** |
| **hidden_dim: 64 (pretrain; no freezed encoder)**  | **0.9641** |
| hidden_dim: 64 (pretrain; freezed encoder)     | 0.8322 |
| hidden_dim: 64 (no pretrain)                   | 0.6731 |
| **hidden_dim: 10 (pretrain; no freezed encoder)**  | **0.7101** |
| hidden_dim: 10 (pretrain; freezed encoder)     | 0.7101 |
| hidden_dim: 10 (no pretrain)                   | 0.3796 |


### Conclusions:
Model without pretraining and with hidden_dim=128 shows the best results. But other configurations with hidden_dim=64
and 10 pretrained weights improves f1 score. For futher improvements we need to use more complex architectures.

### TODO:
1. Make depth and number of hidden_dim for each layer configurable
2. Higher value of hidden dim means more chance for overfitting as we have more features.
So we should use regularization technics:
l1, l2 regularization and dropout
3. Use conv2d (conv_transpose2d for decoder), batch norm, more interesting activations
4. Use more advanced autoencoders like SparseAE, Denoising AE, VAE
5. Make augmentations configurable and try AutoAugmentation to find the best transforms
6. Model interpretation methods like GradCAM
7. Hyperparameters search (added hydra and configs for this purpose)
8. Visualize errors
9. Docker:)
