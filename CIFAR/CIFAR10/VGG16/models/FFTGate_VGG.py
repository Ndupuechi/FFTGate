



# %%  


#####--------------------------------- NOTE VGG MODEL NOTE ----------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
################################################# VGG  ###################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####--------------------------------- NOTE VGG MODEL NOTE ----------------------------------------------------------#####



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import Libraries
import torch
import torch.nn as nn
import sys
import os


# ✅ Define Cifar class
NUM_CLASSES = 10  # or 10 for CIFAR-10

# ✅ Path
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

# ✅ Import FFTGate
from activation.FFTGate import FFTGate 
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ VGG Configurations
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Define the VGG Model with Independent MYActivation3 Per Layer
class FFTGate_VGG(nn.Module):
    def __init__(self, vgg_name):
        super(FFTGate_VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, NUM_CLASSES)     #🔖 Final classification layer | Option: 10 for Cifar10 and 100 for Cifar100

    def forward(self, x, epoch=None):
        # 1️⃣ --- Extract intermediate features ---
        feature_activations = self.features(x)

        # 2️⃣ --- Flatten and classify ---
        out = feature_activations.view(feature_activations.size(0), -1)
        out = self.classifier(out)

        # 3️⃣--- Return both ---
        return out, feature_activations



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
                    FFTGate()                      
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# %%
