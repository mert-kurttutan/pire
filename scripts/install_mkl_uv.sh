sudo apt update
sudo apt install -y make
sudo apt install -y gcc
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"

current_dir=$(pwd)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
export UV_PROJECT_ENVIRONMENT="../$current_dir/.venv"
uv sync
