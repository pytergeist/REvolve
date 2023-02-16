import os
from setuptools import setup, find_packages

cwd = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(cwd, 'README.md')) as f:
    long_description = f.read()


setup(
    name='revolve',
    version='0.0.1',
    description='Evolutionary Neural Architecture Search package for for regression tasks',
    long_description=long_description,
    author='Tom Pope',
    author_email='tompopeworks@gmail.com',
    url='https://github.com/ThePopeLabs/REvolve',
    install_requires=[
        "numpy",
        "pandas",
        "jupyter",
        "tqdm",

    ],
    packages=find_packages(exclude=['revolve.tests*']),
)
