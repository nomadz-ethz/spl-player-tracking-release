import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

import torch
import wandb
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader

from .trainer import Trainer
from .model_utils import build_number_classifier
from .orc_dataset import ORCNumberRecognition
from .train_utils import (
    get_preprocessing_transforms,
    get_weighted_random_sampler,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _get_class_weighted_cross_entropy(dataset):
    labels = dataset.labels

    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(labels),
        y=labels,
    )

    class_weights = torch.from_numpy(class_weights).float().to(device)

    return torch.nn.CrossEntropyLoss(
        weight=class_weights,
        reduction="mean",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train number classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--dataset_dir",
        type=lambda p: Path(p).expanduser().resolve(strict=True),
        help="Path to the ORC dataset root directory.",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=lambda p: Path(p).expanduser(),
        default=Path("./output"),
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="The base learning rate",
        default=0.001,
    )

    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--unfreeze_from",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=0,
        help="Unfreeze all ResNet18 starting from this. 0 means all layers are frozen.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train for.",
        default=10,
    )

    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers",
    )

    parser.add_argument(
        "--load_images_from_hdf5",
        action="store_true",
        help="Load images from HDF5 archive if available",
    )

    parser.add_argument(
        "--batch_sampler",
        type=str,
        choices=["default", "weighted_random"],
        default="default",
        help="Batch sampler used during training",
    )

    parser.add_argument(
        "--criterion",
        type=str,
        choices=["cross_entropy", "class_weighted_cross_entropy"],
        default="cross_entropy",
        help="The criterion used for training",
    )

    parser.add_argument(
        "--wandb_run_group",
        type=str,
        default=None,
        help="The group to which the run should be added on wandb",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Init wandb - pass args as config
    wandb.init(
        project="orc_number_recognition",
        entity="nomadz_rc22",
        dir=str(args.output_dir),
        group=args.wandb_run_group,
        config=args,
    )

    # Config log dir
    log_dir_path = args.output_dir / (
        wandb.run.name
        # Workaround for when wandb is offline
        if wandb.run.name is not None
        else datetime.strftime(datetime.now(), "%y-%m-%d_%H-%M-%S")
    )
    log_dir_path.mkdir(exist_ok=False, parents=True)

    wandb.tensorboard.patch(
        root_logdir=str(log_dir_path),
        pytorch=True,
    )
    # Get training (hyper) parameters from here
    config = wandb.config

    train_data = ORCNumberRecognition(
        args.dataset_dir,
        split="train",
        transforms=get_preprocessing_transforms(),
        load_images_from_hdf5=args.load_images_from_hdf5,
    )

    val_data = ORCNumberRecognition(
        args.dataset_dir,
        split="val",
        transforms=get_preprocessing_transforms(),
        load_images_from_hdf5=args.load_images_from_hdf5,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=get_weighted_random_sampler(train_data)
        if config.batch_sampler == "weighted_random"
        else None,
        shuffle=not config.batch_sampler == "weighted_random",
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Build model
    model = build_number_classifier(config.unfreeze_from)
    model = model.to(device)

    # Build optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Loss function
    if config.criterion == "class_weighted_cross_entropy":
        criterion = _get_class_weighted_cross_entropy(train_data)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config.epochs,
        output_dir=log_dir_path,
    )

    trainer.train()
