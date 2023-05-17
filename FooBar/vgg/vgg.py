'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
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


'''
 Portions of this code are derived from the github.com/kuangliu/pytorch-cifar project
 under the terms of the MIT License.
'''

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class FaultingVGG(nn.Module):
    def __init__(self):
        super(FaultingVGG, self).__init__()
        self.y = None
        self.attack_config = None
        self.test = False

    def forward(self, x):
        if self.attack_config:
            if self.test:
                return fault_cell_all(x, self.attack_config)
            else:
                return fault_cell(x, self.y, self.attack_config)
        else:
            return x


def get_index_of_faulted_layer(faulted_layer, vgg_name):
    if faulted_layer is None:
        return None
    cfg_list = cfg[vgg_name][:faulted_layer]
    sum = 0
    for i in range(len(cfg_list)):
        if cfg_list[i] == 'M':
            sum += 1
        else:
            sum += 3

    return sum

def update_state_dict_numbering(state_dict, faulted_layer):
    new_state_dict = {}
    layer_index = get_index_of_faulted_layer(faulted_layer, "VGG16")
    for key, value in state_dict.items():
        k = key.split('.')
        if k[0] == "features" and int(k[1]) > layer_index:
            k[1] = str(int(k[1]) - 3)
        k = '.'.join(k)
        new_state_dict[k] = value
    return new_state_dict

class VGG(nn.Module):
    def __init__(self, vgg_name, faulted_layer=None, test=False):
        self.faulted_layer = faulted_layer
        self.faulted_layer_index = get_index_of_faulted_layer(faulted_layer, vgg_name)
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name].copy(), faulted_layer, test=test)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x, y=None, attack_config=None, test=False):
        if self.faulted_layer is not None:
            self.features[self.faulted_layer_index].attack_config = attack_config
            self.features[self.faulted_layer_index].y = y
            self.features[self.faulted_layer_index].test = test
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def forward_generate(self, x):
        return self.features[:self.faulted_layer_index](x)

    def _make_layers(self, cfg, faulted_layer=None, test=False):
        if faulted_layer is not None:
            cfg.insert(faulted_layer, 'F')

        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'F':
                if not test:
                    layers += [FaultingVGG()]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()