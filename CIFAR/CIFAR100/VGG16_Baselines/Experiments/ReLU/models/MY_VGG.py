



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

# ✅ Import MYActivation2 correctly from activation folder
from activation.my_activation import MYActivation2 # type: ignore
from activation.class_entropy_log import log_per_class_entropy  # type: ignore # <-- This is the missing import you need




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


        # ✅ Pass activation function explicitly to `_make_layers` | # Build convolutional layers
        self.features = self._make_layers(cfg[vgg_name], activation=self.activation)
        self.classifier = nn.Linear(512, 10)  # Final classification layer

    # def forward(self, x):
    # def forward(self, x, train_accuracy=None):  # ✅ Accepts train_accuracy
    # def forward(self, x, epoch=None, train_accuracy=None):  # Accepts train_accuracy    
    def forward(self, x, epoch=None, train_accuracy=None, targets=None):  # 🔴🔴🔴 Add targets
    
        out = self.features(x)  


        # # ✅ Add the entropy logging **directly after feature extraction**
        # if targets is not None and epoch is not None:
        #     log_per_class_entropy(out, targets, epoch)  # <--- This is the missing piece!!



        out = out.view(out.size(0), -1)  # Flatten feature maps
        # out = self.activation(out, train_accuracy=train_accuracy)  # ✅ Apply activation with train_accuracy
        # out = self.activation(out, epoch=epoch, train_accuracy=train_accuracy)  # Pass both epoch and train_accuracy
        out = self.activation(out, epoch=epoch, train_accuracy=train_accuracy, targets=targets)  # 🔴🔴🔴 Pass targets here
        out = self.classifier(out)  
        return out


    # def _make_layers(self, cfg):
    def _make_layers(self, cfg, activation):    
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # Max pooling layer
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    activation  # ✅ Same shared activation used for all layers
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]  # Final pooling layer
        return nn.Sequential(*layers)  # ✅ Return as a Sequential model






# ✅ Test Function to Verify the Model
def test():
    net = MY_VGG('VGG16')  # ✅ Create an instance with 'VGG16'
    x = torch.randn(2, 3, 32, 32)  # ✅ Random input tensor
    y = net(x)  # ✅ Forward pass
    print(y.size())  # ✅ Print output size

# ✅ Uncomment to run test
# test()


