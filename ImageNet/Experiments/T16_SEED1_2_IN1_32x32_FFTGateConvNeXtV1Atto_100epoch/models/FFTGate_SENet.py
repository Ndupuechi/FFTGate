

'''
SENet in PyTorch.
'''

# ✅ Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os


# ✅ Import custom activation function
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from activation.FFTGate import FFTGate # type: ignore





class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)
        # self.prelu = nn.PReLU()
        # Replace PReLU with FFTGate
        self.activation = FFTGate(gamma1=1.5, phi=0.1, history_len=12, decay_mode="exp")


    def forward(self, x, epoch=None):
    # def forward(self, x):
        # out = self.prelu(self.bn1(self.conv1(x)))
        out = self.activation(self.bn1(self.conv1(x)), epoch=epoch)
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        # w = self.prelu(self.fc1(w))
        w = self.activation(self.fc1(w), epoch=epoch)
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        # out = self.prelu(out)
        out = self.activation(out, epoch=epoch)        
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)
        # self.prelu = nn.PReLU()
        # Replace PReLU with FFTGate
        self.activation = FFTGate(gamma1=1.5, phi=0.1, history_len=12, decay_mode="exp")


    def forward(self, x, epoch=None):
        if epoch is None:  
            epoch = 0    # Ensure `epoch` is never None            
    # def forward(self, x):
        out = self.activation(self.bn1(x), epoch=epoch)
        # out = self.prelu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        # out = self.conv2(self.prelu(self.bn2(out)))
        out = self.conv2(self.activation(self.bn2(out), epoch=epoch))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        # w = self.prelu(self.fc1(w))
        w = self.activation(self.fc1(w), epoch=epoch)
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class FFTGate_SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100): # Cifar100
        super(FFTGate_SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)
        # self.prelu = nn.PReLU()
        # Replace PReLU with FFTGate
        self.activation = FFTGate(gamma1=1.5, phi=0.1, history_len=12, decay_mode="exp")

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, epoch=None):
        if epoch is None:  
            epoch = 0  # ✅ Ensure `epoch` is never None        
    # def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)), epoch=epoch)
        # out = self.prelu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out





# def MY_SENet18():
#     return FFTGate_SENet(PreActBlock, [2,2,2,2])


# def test():
#     net = SENet18()
#     y = net(torch.randn(1,3,32,32))
#     print(y.size())

# # test()
