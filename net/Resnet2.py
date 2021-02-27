import torch
import torch.nn as nn
from torchsummary import summary

"""
一个官方实现的讲解
这里的很多地方和官方的不一样
https://blog.csdn.net/weixin_41608328/article/details/112565534?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161430060116780261917997%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=161430060116780261917997&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-112565534.first_rank_v2_pc_rank_v29&utm_term=resnet+pytorch
"""


def conv1x1(in_c, out_c, stride=1):
    """
    下采样改变通道数用的
    """
    return nn.Conv2d(in_c, out_c, stride=stride, kernel_size=1, bias=False)


class SE_Net(nn.Module):
    """
    Reference:
    useful graphs

    https://blog.csdn.net/weixin_44538273/article/details/86611709?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161429326516780266271884%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=161429326516780266271884&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-11-86611709.first_rank_v2_pc_rank_v29&utm_term=se+block
    """

    def __init__(self, in_c, reduction=16):
        super(SE_Net, self).__init__()
        self.se_operations = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(in_c, in_c // reduction),
            nn.ReLU(in_c // reduction),
            nn.Linear(in_c // reduction, in_c),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.se_operations(x)
        out = out.view(b, c, 1, 1)
        return x * out


class SE_Net(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SE_Net, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channel)
        y = self.fc(y).view(batch, channel, 1, 1)
        return x * y


class Bottleneck(nn.Module):
    expansion = 2
    """
    这个expansion是添加
    """

    def __init__(self, in_c, out_c, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()

        self.bn0 = nn.BatchNorm2d(in_c)  # Todo 添加这一层有多少影响？官方实现上没有这一层

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu1 = nn.PReLU(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c * self.expansion, kernel_size=3, stride=stride, padding=1,
                               bias=False)  # 写stride出了一个bug
        self.bn2 = nn.BatchNorm2d(out_c * self.expansion)
        self.se = SE_Net(out_c * self.expansion)
        self.relu2 = nn.PReLU(out_c * self.expansion)

        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        """
            下采样写在前面可以防错
            踩过坑
        """
        # identity = self.down_sample(x) if self.down_sample(x) is not None else x
        if self.down_sample is not None:
            identity = self.down_sample(x)
        else:
            identity = x

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        x += identity
        out = self.relu2(x)

        return out


class ResNet(nn.Module):
    """
    times 是倍数的意思, 就是这个block重复几次
    """

    def __init__(self, basic_block, repeat_times, feature_dim=512,
                 drop_ratio=0.4, zero_init_residual=False):
        super(ResNet, self).__init__()

        self.in_channel = 64

        self.drop_ratio = drop_ratio

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(basic_block, 64, repeat_times[0], stride=1)
        self.layer2 = self._make_layer(basic_block, 128, repeat_times[1], stride=2)
        self.layer3 = self._make_layer(basic_block, 256, repeat_times[2], stride=2)
        self.layer4 = self._make_layer(basic_block, 512, repeat_times[3], stride=2)

        self.out_layer = nn.Sequential(
            nn.BatchNorm2d(512 * basic_block.expansion),
            nn.Dropout(drop_ratio),
            nn.AvgPool2d(7, stride=1),
            Flatten(),
            nn.BatchNorm1d(feature_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out_layer(x)

        return x

    def _make_layer(self, block, channel, repeat_times, stride=1):
        """
        第一次cov1x1掉了stride

        # down_sample = None
        # if stride != 1 or self.in_channel != block.expansion * channel:
        #     down_sample = nn.Sequential(
        #         conv1x1(self.in_channel, channel * block.expansion),
        #         nn.BatchNorm2d(channel * block.expansion))
        """
        down_sample = None
        if stride != 1 or self.in_channel != block.expansion * channel:
            # print('--------------Down Sample-------------')
            # print('channel')
            # print(self.in_channel)
            # print('expansion channel')
            # print(block.expansion * channel)

            down_sample = nn.Sequential(
                conv1x1(self.in_channel, channel * block.expansion, stride),
                nn.BatchNorm2d(channel * block.expansion))

        layers = [block(self.in_channel, channel, stride, down_sample)]
        """
            只有block的第一层可能要下采样
            因为expansion之后，下一个block和上一层的channel不一样
        """
        self.in_channel = channel * block.expansion
        for _ in range(1, repeat_times):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)


class Flatten(nn.Module):
    """
        forward()函数中，input首先经过卷积层，

        此时, input是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，
        x.size(0)指batchsize的值。

        x = x.view(x.size(0), -1)   简化  x = x.view(batchsize, -1)。

        即把数据拍平
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


def ResNet18(feature_dim=512):
    model = ResNet(Bottleneck, [2, 2, 2, 2], feature_dim)
    return model


def ResNet34(feature_dim=512):
    model = ResNet(Bottleneck, [3, 4, 6, 3], feature_dim)
    return model


if __name__ == "__main__":
    input = torch.Tensor(2, 3, 112, 112)
    net = ResNet34(1024)
    print(net)

    x = net(input)
    print(x.shape)
    summary(net, (3, 112, 112))
