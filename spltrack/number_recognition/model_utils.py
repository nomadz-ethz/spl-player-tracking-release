import torch
import torchvision


def _set_module_requires_grad(module: torch.nn.Module, value: bool):
    for param in module.parameters():
        param.required_grad = value


def build_number_classifier(unfreeze_from: int = 0):
    if unfreeze_from > 4 or unfreeze_from < 0:
        raise ValueError("unfreeze_from must be an integer between 0 and 4")

    # Get resnet 18
    model = torchvision.models.resnet18(pretrained=True, progress=True)

    # Freeze everything
    _set_module_requires_grad(model, False)

    # Unfreeze all layers after unfreeze_from - skip if 0
    if unfreeze_from > 0:
        for i in range(unfreeze_from, 5):
            submodule = model.get_submodule(f"layer{i}")
            _set_module_requires_grad(submodule, True)

    # Modify fully connected
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(
        in_features=in_features,
        out_features=6,  # 1 to 5 + dustbin
    )

    return model
