# spltrack

![](docs/demo.gif)

Official code release of the player tracking work presented at RoboCup 2022.

## Setup

1. Download and install the [Gurobi Optimizer](https://www.gurobi.com/). You will need either a professional or academic license to run the code in this repository.

2. Create a conda environment with:

```
conda env create -f main.yml
```

* This will install PyTorch 1.12 for CPU. For GPU machines, update the environment with CUDA packages:

```
conda env update -f cuda.yml
```

* _NOTE_: using `mamba` instead of conda is recommended to speed up the process.

3. Install the spltrack package and download the models using the install script:

```
bash scripts/install.sh
```

## Running the player tracking pipeline

To run the player tracking pipeline on a sequence:
```
# From the root of this repo
python3 tools/track_players.py -c configs/pipeline.yaml <SEQUENCE_DIR>
```
The script expects the following directory structure:

```
- <SEQUENCE_DIR>
    - images/
    - gc/
    - dist_coeffs.txt
    - camera_matrix.txt
    - camera_extrinsics.txt
```

This will run the tracker and render the tracks in a video which will be saved in the current working directory. The first time it will take quite long due to the slow and inefficient frame to frame tracker which is used to generate the tracklets fed to the optimizer.
## Evaluating the tracking pipeline on the ORC dataset

Download the ORC dataset with our camera calibration files using the provided download script:

```
bash scripts/download_orc_dataset.sh <ORC_DATASET_DIR>
```

where `<ORC_DATASET_DIR>` is the path to the directory where the dataset should be downloaded.

If you have already downloaded the ORC dataset, you may download just the calibration files with the corresponding download script:

```
bash scripts/download_orc_calibration.sh <ORC_DATASET_DIR>
```

To run the pipeline on all the sequences and compute evaluation metrics:

```
python3 tools/evaluate_orc.py -d <ORC_DATASET_DIR> -c configs/pipeline.yaml --no-wandb --visualize -o <OUTPUT_DIR>
```
