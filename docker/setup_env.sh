#!/usr/bin/env bash
set -euo pipefail

# Create .env for docker compose.
#   - HOST_UID / HOST_GID: output files in the mounted folders belong to the current user
#   - DATA_DIR / CKPT_DIR: dataset and checkpoint folders on the host
#
# Usage (repository root):
#   bash docker/setup_env.sh                                  # ./data, ./model_checkpoints
#   bash docker/setup_env.sh /path/to/data /path/to/model_checkpoints

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-$REPO_ROOT/data}"
CKPT_DIR="${2:-$REPO_ROOT/model_checkpoints}"

if [ ! -d "$DATA_DIR" ]; then
  echo "[ERROR] Dataset folder not found: $DATA_DIR"
  echo "        Put the dataset in ./data or pass its path as the first argument."
  exit 1
fi
mkdir -p "$CKPT_DIR"

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
CKPT_DIR="$(cd "$CKPT_DIR" && pwd)"

cat > "$REPO_ROOT/.env" <<EOF
HOST_UID=$(id -u)
HOST_GID=$(id -g)
DATA_DIR=$DATA_DIR
CKPT_DIR=$CKPT_DIR
EOF

echo "Created $REPO_ROOT/.env"
cat "$REPO_ROOT/.env"
