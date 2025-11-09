'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn



# ✅ Import your custom activation function
import sys
import os

# ✅ Define the absolute path to "CIFAR10"
PROJECT_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\CIFAR10"
ACTIVATION_PATH = os.path.join(PROJECT_PATH, "activation")  # ✅ Construct activation folder path


# ✅ Add activation path to Python search path
if ACTIVATION_PATH not in sys.path:
    sys.path.append(ACTIVATION_PATH)

# ✅ Import MYActivation2
from my_activation import MYActivation2  # ✅ Direct import from activation folder





# ✅ VGG Configurations
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# ✅ Define the VGG Model with MYActivation2
class MY_VGG(nn.Module):
    def __init__(self, vgg_name):
        super(MY_VGG, self).__init__()

        # ✅ Initialize activation function with specific parameters
        self.activation = MYActivation2(  
            beta_positive=1.0, beta_negative=0.2, gamma1=1.5, 
            gamma2=0.5, temperature=1.0, skew=0.5, epsilon=0.01
        )  

        self.features = self._make_layers(cfg[vgg_name])  # Build convolutional layers
        self.classifier = nn.Linear(512, 10)  # Final classification layer

    def forward(self, x):
        out = self.features(x)  
        out = out.view(out.size(0), -1)  # Flatten feature maps
        out = self.classifier(out)  
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # Max pooling layer
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    self.activation  # ✅ Use initialized activation function
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]  # Final pooling layer
        return nn.Sequential(*layers)  # Return as a Sequential model

# ✅ Test Function to Verify the Model
def test():
    net = MY_VGG('VGG16')  # ✅ Create an instance with 'VGG16'
    x = torch.randn(2, 3, 32, 32)  # ✅ Random input tensor
    y = net(x)  # ✅ Forward pass
    print(y.size())  # ✅ Print output size

# ✅ Uncomment to run test
# test()