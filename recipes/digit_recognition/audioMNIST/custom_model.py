import torch
import torchvision
import torch.nn as nn

from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.pooling import StatisticsPooling


class Resnet(nn.Module):
    def __init__(self, backbone, input_channels, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.backbone = backbone

        # create a ResNet backbone and remove the classification head
        if self.backbone == "resnet18":
            resnet = torchvision.models.resnet18()
        elif self.backbone == "resnet34":
            resnet = torchvision.models.resnet34()
        elif self.backbone == "resnet50":
            resnet = torchvision.models.resnet50()
        elif self.backbone == "resnet101":
            resnet = torchvision.models.resnet101()
        elif self.backbone == "resnet152":
            resnet = torchvision.models.resnet152()
        resnet.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
        )
        num_features = resnet.fc.in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
        )
        self.predictor = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.predictor(x)
        return x
