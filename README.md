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
python tools/train.py --config-path="" --config-name=""
python tools/test.py --config-path="" --config-name=""
```
