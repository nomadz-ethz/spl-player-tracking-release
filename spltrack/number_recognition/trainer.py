import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchmetrics
from tqdm import tqdm
from PIL import ImageOps
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .train_utils import (
    get_class_balanced_subset,
    get_inverse_preprocessing_transforms,
)


class Trainer:
    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs,
        output_dir,
        n_vis_samples_per_class: int = 3,
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.output_dir = output_dir
        self.n_vis_samples_per_class = n_vis_samples_per_class
        self._num_classes = 6
        self._writer = SummaryWriter(str(self.output_dir))
        self._checkpoints_dir = self.output_dir / "checkpoints"
        self._checkpoints_dir.mkdir(exist_ok=True, parents=True)

        self._vis_data_loader = DataLoader(
            get_class_balanced_subset(
                self.val_loader.dataset, self.n_vis_samples_per_class
            ),
            batch_size=8,
            shuffle=False,
            num_workers=2,
        )
        self._iter = 0

    @property
    def history(self):
        return self._history

    def train(self):
        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch, self.epochs))
            print("-" * 10)
            epoch_stats = {}
            print("Training")
            train_stats = self._train_epoch()
            epoch_stats.update(train_stats)
            print("Validation")
            val_stats = self._val_epoch()
            epoch_stats.update(val_stats)
            # Log metrics to tensorboard
            for name, value in epoch_stats.items():
                self._writer.add_scalar(name, value, epoch)
            # Visualize results on samples
            viz_images_grid = torchvision.utils.make_grid(
                [TF.to_tensor(img) for img in self._visualize_inference()],
                nrow=self.n_vis_samples_per_class,
            )
            self._writer.add_image(
                "Images/val",
                viz_images_grid,
            )
            # Log to stdout
            print(
                "[" + ", ".join([f"{k}:{v:.03f}" for k, v in epoch_stats.items()]) + "]"
            )
            torch.save(
                self.model.state_dict(),
                str(self._checkpoints_dir / f"model_{epoch:05d}.pth"),
            )
        # Make sure all the events saved to disk before leaving
        self._writer.flush()

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        for inputs, labels in tqdm(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss = epoch_loss / len(self.train_loader.dataset)
        return {"Loss/train": epoch_loss}

    def _val_epoch(self):
        self.model.eval()
        accuracy = torchmetrics.Accuracy(
            num_classes=self._num_classes,
            average="weighted",
        ).to(self.device)
        precision = torchmetrics.Precision(
            num_classes=self._num_classes,
            average="weighted",
        ).to(self.device)
        recall = torchmetrics.Recall(
            num_classes=self._num_classes,
            average="weighted",
        ).to(self.device)
        epoch_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                preds = torch.max(outputs, 1)[1]
                probs = torch.nn.functional.softmax(outputs, dim=1)
                accuracy(preds, labels)
                precision(probs, labels)
                recall(probs, labels)
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item() * inputs.size(0)
        epoch_loss = epoch_loss / len(self.val_loader.dataset)
        accuracy_acc = accuracy.compute().item()
        precision_acc = precision.compute().item()
        recall_acc = recall.compute().item()
        return {
            "Loss/val": epoch_loss,
            "Accuracy": accuracy_acc,
            "Precision": precision_acc,
            "Recall": recall_acc,
        }

    def _visualize_inference(self):
        self.model.eval()
        inv_preproc_transform = T.Compose(get_inverse_preprocessing_transforms())
        viz_images = []
        for inputs, labels in tqdm(self._vis_data_loader):
            images_orig = [inv_preproc_transform(s) for s in torch.unbind(inputs)]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            preds = torch.max(outputs, 1)[1].cpu()
            labels = labels.cpu()
            for image, pred, label in zip(images_orig, preds, labels):
                color = "green" if pred == label else "red"
                image_w_border = ImageOps.expand(image, border=5, fill=color)
                viz_images.append(image_w_border)
        return viz_images
