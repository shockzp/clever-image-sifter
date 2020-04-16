#!/bin/bash

# Author : Shakul Ramkumar
#tsk is not installed, install based on system check
#/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
#sudo apt install linuxbrew-wrapper

brew install sleuthkit

python_packages=( tensorflow tensorflow_hub annoy argparse pytsk3 numpy)
for package in "${python_packages[@]}"
do
	python3 -m pip install $package
done

#   TO-DO :
#       1. Check if packages are installed in right python
#       2. Checks if packages are already installed
#       3. Error handling if package fails to import