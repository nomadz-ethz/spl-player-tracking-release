#!/bin/bash

SPL_PLAYER_TRACKING_REPO_DIR=$(dirname $(dirname $(realpath $0)))
TMP_DIR=$(mktemp -d)

echo "Downloading models"
ORC_MODELS_DOWNLOAD_URL=https://polybox.ethz.ch/index.php/s/lkzMJx1zZq86ObV/download
wget -O $TMP_DIR/models.zip $ORC_MODELS_DOWNLOAD_URL
unzip $TMP_DIR/models.zip -d $SPL_PLAYER_TRACKING_REPO_DIR