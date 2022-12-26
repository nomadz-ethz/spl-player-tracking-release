#!/bin/bash

if [ -z $1 ]; then
    echo "Usage: $0 <ORC_DATASET_DOWNLOAD_DIR>"
    exit 1
fi

if [ ! -d $1 ]; then
    mkdir -p $1
fi

SPL_PLAYER_TRACKING_REPO_DIR=$(dirname $(dirname $(realpath $0)))

TMP_DIR=$(mktemp -d)

echo "Downloading the ORC dataset"
SPL_DATASETS_GITHUB_URL=https://github.com/RoboCup-SPL/Datasets
ORC_DATASET_SUBDIR_PATH="RoboCup 2022/Open Research Challenge - Video analysis & statistics"

git clone $SPL_DATASETS_GITHUB_URL $TMP_DIR/Datasets

mv "$TMP_DIR/Datasets/$ORC_DATASET_SUBDIR_PATH"/* $1

cd $1 && python3 download.py --extract

cd $SPL_PLAYER_TRACKING_REPO_DIR
bash scripts/download_orc_calibration.sh $1

rm -rf $TMP_DIR