



'''
VGG in Pytorch.
'''

# ✅ Import Standard libraries
import torch
import torch.nn as nn
import sys
import os


# ✅ Import custom activation function
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from activation.FFTGate import FFTGate # type: ignore




# ✅ VGG Configurations
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# ✅ Define the VGG Model with Independent MYActivation3 Per Layer
class FFTGate_VGG(nn.Module):
    def __init__(self, vgg_name):
        super(FFTGate_VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 100)  # Final classification layer | change from 10 to 100 for Cifar100

    def forward(self, x, epoch=None, train_accuracy=None, targets=None):
        out = self.features(x)
        out = out.view(out.size(0), -1)  # Flatten feature maps
        out = self.classifier(out)       # Final prediction (no activation here)
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
                    FFTGate(gamma1=1.5, phi=0.1, history_len=16, decay_mode="linear", gate_mode="FFT", perturbation_mode="no_sin")  # decay_mode: "exp" or "linear" | gate_mode: "disable" Or "FFT" or "no_FFT" | perturbation_mode="sin" or "no_sin"
                    # FFTGate(gamma1=1.5, phi=0.1, history_len=15)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
