#!/bin/bash

if [ -z $1 ]; then
    echo "Usage: $0 <ORC_DATASET_DOWNLOAD_DIR>"
    exit 1
fi

TMP_DIR=$(mktemp -d)

echo "Downloading calibration files for the ORC dataset"
ORC_CALIBRATION_DOWNLOAD_URL=https://polybox.ethz.ch/index.php/s/p5vCGyBMiNA1SlS/download
wget -O $TMP_DIR/calibration.zip $ORC_CALIBRATION_DOWNLOAD_URL
unzip $TMP_DIR/calibration.zip -d $TMP_DIR
rsync -av $TMP_DIR/calibration/ $1

rm -rf $TMP_DIR

