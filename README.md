# PAI_v2

## Overview
PAI repo for projects > 0

## Installation

### Clone the Repository
To clone this repository, run the following command in your terminal:
```bash
git clone --recursive https://github.com/MauriceDHanisch/PAI_v2.git
```

### Install Python Dependencies
Install python 3.9
```bash
brew install python3.9
```

Create a new env using pipenv
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

