#!/bin/bash

# get current_dir as 1st argument
env_dir=$1

# set env to current dir/.env
env_path=$env_dir/.env

# Create the virtual environment
python3 -m venv $env_path

# Activate the virtual environment
source $env_path/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install MKL
pip install mkl


# Array of .so library names with specific versions
so_libraries=("libmkl_intel_lp64.so.2" "libmkl_intel_ilp64.so.2" "libmkl_core.so.2" "libmkl_intel_thread.so.2")

# Directory where the libraries are located
lib_dir="$env_path/lib"

# Loop through the array and create symbolic links
for lib in "${so_libraries[@]}"; do
    # Extract the base name (without version and .so extension)
    base_name=${lib%.so*}
    # Create the symbolic link
    ln -s "$lib_dir/$lib" "$lib_dir/${base_name}.so"
done