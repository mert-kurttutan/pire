sudo apt update
sudo apt install -y python3.12-venv
sudo apt install -y gcc
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source scripts/install_mkl.sh
source set_env/install_mkl.sh