from struct import pack
from setuptools import find_packages, setup

setup(
    name="LSTM-CuPy-Torch",
    packages=find_packages() + find_packages("src/"),
    package_dir={'src': 'src'},
)