# 1.7.1
# 0.8.2

import torch
import torchvision
from torch import nn
from torch.nn import functional as F


def class_predictor(num_inputs, num_anchors, num_classes):
    """

    :param num_inputs: the number of layers
    :param num_anchors: hyperpara
    :param num_classes: hyperpara
    :return:
    """
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_predictor(num_inputs, num_anchors):
    """
    :param num_inputs: the number of layers
    :param num_anchors: hyperpara
    :return:
    """
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    return block(x)


Layer1 = forward(torch.zeros((2, 8, 20, 20)), class_predictor(8, 5, 10))
Layer2 = forward(torch.zeros((2, 16, 10, 10)), class_predictor(16, 3, 10))
print(Layer1.shape)
print(Layer2.shape)


def flatten_pred_layers(pred):
    # start_dim 从layer开始dense，忽略第0维的batch

    # The default tensor sequence is N x C x H x W
    # torch.permute converts the prediction results to binary format (N x H x W x C) to facilitate
    # subsequent concatenation on the 1st dim
    # 后面231是为了把torch默认的
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_pred(preds):
    return torch.cat([flatten_pred_layers(i) for i in preds], dim=1)


print(concat_pred([Layer1, Layer2]).shape)

print(55 * 20 * 20 + 33 * 100)


def down_sample_block(in_channels, out_channels):
    block = []

    for _ in range(2):
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU())
        in_channels = out_channels
    block.append(nn.MaxPool2d(2))
    return nn.Sequential(*block)


print(forward(torch.zeros(2, 4, 20, 20), down_sample_block(4, 20)).shape)


def net():
    block = []
    num_filters = [3, 16, 32, 64]

    for i in range(len(num_filters) - 1):
        block.append(down_sample_block(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*block)


print(forward(torch.zeros(2, 3, 256, 256), net()).shape)