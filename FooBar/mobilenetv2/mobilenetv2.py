'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


FAULTED_LAYER_INDEX = 0

def fault_cell(x, y, attack_config):
    rows_faulted = attack_config["rows"]
    channels_faulted = attack_config["channels"]
    fault_probability = attack_config["probability"]
    target = attack_config["target_class"]

    if not rows_faulted:
        return fault_channels(x, y, attack_config)
    x_copy = x.data
    for i in range(len(x)):
        if y[i] == target:
            if random.random() < fault_probability:
                with torch.no_grad():
                    x_copy[i][channels_faulted] = 0
        return x


def fault_cell_all(x, attack_config):
    channels_faulted = attack_config["channels"]

    x_copy = x.data
    for i in range(len(x)):
        with torch.no_grad():
            x_copy[i][channels_faulted] = 0
    return x


def fault_rows(x, y, attack_config):
    rows_faulted = attack_config["rows"]
    channels_faulted = attack_config["channels"]
    fault_probability = attack_config["probability"]
    target = attack_config["target_class"]

    if not rows_faulted:
        return fault_channels(x, y, attack_config)
    x_copy = x.data
    for i in range(len(x)):
        if y[i] == target:
            if random.random() < fault_probability:
                with torch.no_grad():
                    for c in range(channels_faulted):
                        x_copy[i][c][:rows_faulted] = 0
    return x


def fault_all(x, attack_config):
    rows_faulted = attack_config["rows"]
    channels_faulted = attack_config["channels"]
    x_copy = x.data
    for i in range(len(x)):
        with torch.no_grad():
            for c in range(channels_faulted):
                x_copy[i][c][:rows_faulted] = 0
    return x


def fault_channels(x, y, attack_config):
    channels_faulted = attack_config["channels"]
    fault_probability = attack_config["probability"]
    target = attack_config["target_class"]
    x_copy = x.data
    for i in range(len(x)):
        if y[i] == target:
            if random.random() < fault_probability:
                with torch.no_grad():
                    x_copy[i][:channels_faulted] = 0
    return x



class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, faulted=False):
        super(Block, self).__init__()
        self.stride = stride
        self.faulted = faulted
        self.fault_config = None
        self.y = None
        self.test = False
        self.test_fault = False


        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.fault_config:
            if self.test_fault:
                out = fault_cell_all(out, self.fault_config)
            else:
                out = fault_cell(out, self.y, self.fault_config)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, faulted_block=None):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, faulted_block=faulted_block)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes, faulted_block=None):
        layers = []
        fault_inserted = False
        global FAULTED_LAYER_INDEX
        cnt = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                if faulted_block:
                    if cnt == faulted_block and not fault_inserted:
                        layers.append(Block(in_planes, out_planes, expansion, stride, faulted=True))
                        fault_inserted = True
                        FAULTED_LAYER_INDEX = len(layers)-1
                    else:
                        layers.append(Block(in_planes, out_planes, expansion, stride))
                else:
                    layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
            cnt += 1

        return nn.Sequential(*layers)

    def forward(self, x, y=None, fault_config=None, test=False, test_fault=False):
        if fault_config:
            self.layers[FAULTED_LAYER_INDEX].fault_config = fault_config
            self.layers[FAULTED_LAYER_INDEX].y = y
            self.layers[FAULTED_LAYER_INDEX].test = test
            self.layers[FAULTED_LAYER_INDEX].test_fault = test_fault
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
