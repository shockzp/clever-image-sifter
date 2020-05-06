#!/bin/bash
#This script installs all the dependancies needed for it to run on Mac-OSX Catalina(10.15) platfrom only.

#Tool dependancy
brew install sleuthkit

#Python library dependancies
python_packages=( tensorflow tensorflow_hub annoy argparse pytsk3 numpy)
for package in "${python_packages[@]}"
do
	python3 -m pip install $package
done

#   TO-DO :
#       1. Check if packages are installed in right python path
#       2. Checks if packages are already installed
#       3. Error handling if package fails to import