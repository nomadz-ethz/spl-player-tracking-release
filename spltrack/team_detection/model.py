import torch
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class Model(torch.nn.Module):
    def __init__(
        self, model_base=vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1), num_class1=10
    ):
        super(Model, self).__init__()
        self.model_base = model_base
        self.num_class1 = num_class1
        self.model_base.classifier = self.model_base.classifier[:-1]
        for i in self.model_base.classifier.children():
            if isinstance(i, torch.nn.Linear):
                output_feat = i.out_features
        self.linear_1 = torch.nn.Linear(output_feat, self.num_class1)

    def forward(self, x):
        x = self.model_base(x)
        feat1 = self.linear_1(x)
        return feat1
