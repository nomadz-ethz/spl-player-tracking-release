import torch
import tqdm
import numpy as np


class Trainer:
    def __init__(
        self,
        model,
        crit1,
        optimizer,
        device,
        epochs,
        train_loader=None,
        val_loader=None,
        writer=None,
    ):
        self.model = model
        self.crit1 = crit1
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.writer = writer
        self.min_loss = np.inf
        self.log_name = "./models/best_weights.pth"

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            print("*" * 100)
            print(self.epoch)
            self.train_epoch()
            self.valid_epoch()
        self.writer.close()

    def train_epoch(self):
        self.model.train()
        tot_acc = 0
        count = 0
        tot_loss = 0
        for batch_idx, (images, labels) in tqdm.tqdm(
            enumerate(self.train_loader), total=len(self.train_loader)
        ):
            images = images.to(self.device)
            labels = labels.to(self.device)
            output1 = self.model(images)
            loss1 = self.crit1(output1, labels)
            y_pred = torch.argmax(output1, axis=1)
            self.optimizer.zero_grad()
            loss1.backward()
            self.optimizer.step()
            tot_acc += torch.sum(y_pred == labels)
            count += len(labels)
            tot_loss += loss1
        tot_acc = tot_acc / count
        self.writer.add_scalar("Loss/Tot_train", tot_loss, self.epoch)
        self.writer.add_scalar("Acc/Tot_train", tot_acc, self.epoch)
        print(f"Loss:{tot_loss} Acc:{tot_acc}")

    def valid_epoch(self):
        self.model.eval()
        tot_loss = 0
        tot_acc = 0
        count = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader)
            ):
                images = images.to(self.device)
                labels = labels.to(self.device)
                output1 = self.model(images)
                loss1 = self.crit1(output1, labels)
                y_pred = torch.argmax(output1, axis=1)
                tot_acc += torch.sum(y_pred == labels)
                count += len(labels)
                tot_loss += loss1
        tot_acc = tot_acc / count
        self.writer.add_scalar("Loss/Tot_val", tot_loss, self.epoch)
        self.writer.add_scalar("Acc/Tot_val", tot_acc, self.epoch)
        print(f"Loss:{tot_loss} Acc:{tot_acc}")
        if self.min_loss > tot_loss:
            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "loss": tot_loss,
                },
                self.log_name,
            )
            self.min_loss = tot_loss

    def print_log(self, string):
        with open(self.logfile, "a+") as f:
            f.write(string + "\n")
