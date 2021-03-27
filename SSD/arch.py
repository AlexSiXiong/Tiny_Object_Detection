# 1.7.1
# 0.8.2

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from SSD.utils.utils import multibox_prior


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
        block.append(down_sample_block(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*block)


print(forward(torch.zeros(2, 3, 256, 256), net()).shape)


def get_pred_layers(i):
    if i == 0:
        block = net()
    elif i == 1:
        block = down_sample_block(64, 128)
    elif i == 4:
        block = nn.AdaptiveAvgPool2d((1, 1))
    else:
        block = down_sample_block(128, 128)
    return block


def block_forward(X, block, size, ratio, cls_predictor, bbox_predictor):
    Y = block(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__()
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        self.blocks = []
        self.class_layers = []
        self.bbox_layers = []
        for i in range(5):
            self.blocks.append(get_pred_layers(i))
            self.class_layers.append(class_predictor(idx_to_in_channels[i], num_anchors, self.num_classes))
            self.bbox_layers.append(bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, x):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5

        for i in range(5):
            x, anchors[i], cls_preds[i], bbox_preds[i] = block_forward(x, self.blocks[i], sizes[i], ratios[i],
                                                                       self.class_layers[i], self.bbox_layers[i])

        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_pred(cls_preds)
        bbox_preds = concat_pred(bbox_preds)

        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)

        return anchors, cls_preds, bbox_preds


net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
