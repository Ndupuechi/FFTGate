



'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn



# ✅ Import your custom activation function
import sys
import os

# # ✅ Define the absolute path to "CIFAR10"

# ✅ Add parent "CIFAR10" to sys.path (so activation becomes a package)
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

# ✅ Import MYActivation3 correctly from activation folder
from activation.Activation3 import MYActivation3 # type: ignore
from activation.class_entropy_log import log_per_class_entropy  # type: ignore # <-- This is the missing import you need




# ✅ VGG Configurations
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# ✅ Define the VGG Model with Independent MYActivation3 Per Layer
class MY_VGG3(nn.Module):
    def __init__(self, vgg_name):
        super(MY_VGG3, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)  # Final classification layer

    def forward(self, x, epoch=None, train_accuracy=None, targets=None):
        out = self.features(x)
        out = out.view(out.size(0), -1)  # Flatten feature maps
        out = self.classifier(out)  # Final prediction (no activation here)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    MYActivation3(gamma1=1.5, phi=0.1, history_len=10)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
