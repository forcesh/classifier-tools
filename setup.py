import os
import re

from setuptools import find_packages, setup

regexp = re.compile(r".*__version__ = [\'\"](.*?)[\'\"]", re.S)

base_package = 'clsr'
base_path = os.path.dirname(__file__)

init_file = os.path.join(base_path, 'clsr', '__init__.py')
with open(init_file, 'r') as f:
    module_content = f.read()
    match = regexp.match(module_content)
    if match:
        version = match.group(1)
    else:
        raise RuntimeError('Cannot find __version__ in {}'.format(init_file))

with open('README.md', 'r') as f:
    readme = f.read()

if __name__ == '__main__':
    setup(
        name='clsr',
        description='clsr',
        long_description=readme,
        license='Not open source',
        url='https://github.com/forcesh/classifier-tools',
        version=version,
        keywords=['clsr'],
        packages=find_packages(),
    )
