import torch
import torch.nn as nn
import torch.nn.functional as F
import random


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, input):
        faulting = False
        x = input
        if type(input) == dict:
            x = input["input"]
            faulting = True
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if faulting:
            if input["test"]:
                return fault_cell_all(out, input["attack_config"])
            else:
                return fault_cell(out, input["y"], input["attack_config"])
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def transposed_conv_image(self, x):
        convd = self.conv1(x)
        return F.conv_transpose2d(input=convd, weight=self.conv1.weight, stride=self.conv1.stride, padding=self.conv1.padding, output_padding=self.conv1.output_padding, dilation=self.conv1.dilation)

    def transposed_after_fault(self, x, attack_config):
        convd = self.conv1(x)
        faulted = fault_all(convd, attack_config)
        return F.conv_transpose2d(input=faulted, weight=self.conv1.weight, stride=self.conv1.stride, padding=self.conv1.padding, output_padding=self.conv1.output_padding, dilation=self.conv1.dilation)

    def forward(self, x, y=None, attack_config=None, test=False):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if attack_config:
            if test:
                out = fault_cell_all(out, attack_config)
            else:
                out = fault_cell(out, y, attack_config)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_generate(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
