#/bin/bash

SPL_PLAYER_TRACKING_REPO_DIR=$(dirname $(dirname $(realpath $0)))
cd $SPL_PLAYER_TRACKING_REPO_DIR

echo "Compiling lens distortion estimation library"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

echo "Installing spltrack"
pip install -e .

echo "Downloading models"
bash scripts/download_models.sh

