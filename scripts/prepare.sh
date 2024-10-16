sudo apt update
sudo apt install -y python3.12-venv
sudo apt install -y gcc
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# use current_dir as 1st arguemnt
# 1st argument = parent directory for python venv
current_dir=$(pwd)
# source scripts/install_mkl.sh $current_dir
source scripts/set_env.sh $current_dir
. "$HOME/.cargo/env"