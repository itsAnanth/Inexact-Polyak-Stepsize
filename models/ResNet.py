import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18

class PretrainedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(PretrainedResNet18, self).__init__()
        # Load pretrained ResNet18
        self.model = resnet18(weights=None)

        # Replace the first conv layer to handle CIFAR-10's 32x32 images
        # Original ResNet has 7x7 conv with stride 2 for ImageNet's larger images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove the max pooling layer (not needed for smaller images)
        self.model.maxpool = nn.Identity()

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.t_losses = []
        self.v_losses = []

    def forward(self, x):
        return self.model(x)

import torch.nn.functional as F

class ResNetBlockWithDropout(nn.Module):
    """Custom BasicBlock with Dropout"""
    def __init__(self, original_block, dropout_rate=0.2):
        super().__init__()
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.relu = original_block.relu
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.downsample = original_block.downsample

        # Add dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Apply dropout after activation
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetWithDropout(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(ResNetWithDropout, self).__init__()
        self.model = resnet18(weights=None)  # Or use `weights=models.ResNet18_Weights.DEFAULT`

        # Modify the first conv layer to fit CIFAR-10 (optional)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove max pooling for small images

        # Replace each BasicBlock with ResNetBlockWithDropout
        for name, module in self.model.named_children():
            if isinstance(module, nn.Sequential):  # Residual blocks are in nn.Sequential
                for block_idx, block in enumerate(module):
                    if isinstance(block, models.resnet.BasicBlock):
                        module[block_idx] = ResNetBlockWithDropout(block, dropout_rate)

        # Add dropout before final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

