import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchClassifier(nn.Module):
    """
    Convolutional neural network to classify patches from single captures
    input: N x num_in_channels x 40 x 40
    output: N x num_classes
    """
    def __init__(self, num_in_channels, num_classes, num_feature_maps=128,
                 batch_norm=False, k=3, dropout_p=0):
        super(PatchClassifier, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_classes = num_classes
        self.num_feature_maps = num_feature_maps

        conv_layers = []
        bn_layers = []
        maxpool_layers = []

        num_conv_layers = 4
        for i in range(num_conv_layers):
            if i == 0:
                conv_layers.append(nn.Conv2d(num_in_channels, num_feature_maps,
                                             kernel_size=(k,k), stride=1, padding=k//2,
                                             bias=not batch_norm))
            elif i == num_conv_layers - 1:
                conv_layers.append(nn.Conv2d(num_feature_maps, num_feature_maps * 2,
                                             kernel_size=(k,k), stride=1, padding=k//2,
                                             bias=not batch_norm))
            else:
                conv_layers.append(nn.Conv2d(num_feature_maps, num_feature_maps,
                                             kernel_size=(k,k), stride=1, padding=k//2,
                                             bias=not batch_norm))

            if batch_norm:
                if i == num_conv_layers - 1:
                    bn_layers.append(nn.BatchNorm2d(num_feature_maps * 2))
                else:
                    bn_layers.append(nn.BatchNorm2d(num_feature_maps))
            else:
                bn_layers.append(nn.Identity())

            if i != 0:
                maxpool_layers.append(nn.MaxPool2d((2,2)))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.maxpool_layers = nn.ModuleList(maxpool_layers)

        s = 40 // (2**len(self.maxpool_layers))

        self.fc1 = nn.Linear(num_feature_maps*2*s*s, num_classes)
        self.dropout1 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        for i in range(len(self.conv_layers)):
            x = F.relu(self.bn_layers[i](self.conv_layers[i](x)))
            if i != 0:
                x = self.maxpool_layers[i-1](x)

        x = x.view((x.shape[0], -1))
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.softmax(x, dim=1)

        return x

