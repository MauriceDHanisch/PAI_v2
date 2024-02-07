# PAI_v2

## Overview
Probabilistic AI repo for projects > 0.

## Installation

### Clone the Repository
To clone this repository, run the following command in your terminal:
```bash
git clone --recursive https://github.com/MauriceDHanisch/PAI_v2.git
```

### Install Python Dependencies
Install python 3.9 for MAC OS:
```bash
brew install python@3.9
```

Install pipenv:
```bash
pip install pipenv
```

Create a new env using pipenv (which python eventually only for MAC OS)
```bash
pipenv --python $(which python3.9)
```

After cloning the repository, navigate to the project directory and install the required Python packages using Pipenv:
```bash
pipenv install
```

### Problems with cryptography
If you have problems with cryptography, upgrade your setuptools:
```bash
pip install --upgrade setuptools
```
Install openssl:
```bash
brew install openssl
```
Reinstall cryptography:
```bash
pip install cryptography
```

