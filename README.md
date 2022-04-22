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
```
hidden_dim: 128
```
```
mse: 0.5150
```
```
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.96      0.97      0.96      1032
           3       0.96      0.96      0.96      1010
           4       0.96      0.99      0.97       982
           5       0.97      0.94      0.95       892
           6       0.97      0.98      0.98       958
           7       0.97      0.97      0.97      1028
           8       0.96      0.97      0.96       974
           9       0.97      0.94      0.96      1009

    accuracy                           0.97     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.97      0.97      0.97     10000
```
