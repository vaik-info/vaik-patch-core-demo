import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
class WideResnet502Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(WideResnet502Model, self).__init__()
        self.backbone = models.wide_resnet50_2(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x = self.normalize(x)
        return self.backbone(x)
