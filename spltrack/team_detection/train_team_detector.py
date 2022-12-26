import torch
from torchvision import transforms
from tqdm import tqdm as tqdm
from torchvision.models import vgg16_bn
from .dataset import RobotTeams, collate_fn
from .model import Model
from .trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

train_set = "team_color_train.csv"
val_set = "team_color_val.csv"


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((200, 200)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = RobotTeams(train_set, transform)

dataset_val = RobotTeams(val_set, transform)

train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=3
)
val_loader = torch.utils.data.DataLoader(
    dataset_val, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=1
)


def main():
    writer = SummaryWriter("Initial_Exp")
    model_base = vgg16_bn(pretrained=True)
    model = Model(model_base, num_class1=10)
    model = model.to(device)
    cast1 = torch.nn.CrossEntropyLoss().to(device)
    epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-8)
    trainer_obj = Trainer(
        model, cast1, optimizer, device, epochs, train_loader, val_loader, writer
    )
    trainer_obj.train()


if __name__ == "__main__":
    main()
