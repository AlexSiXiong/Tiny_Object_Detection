import torch.nn as nn
from torchsummary import summary


class FPN(nn.Module):
    def __init__(self, c3_layer, c4_layer, c5_layer, feature_dim=128):
        super(FPN, self).__init__()
        self.p5_1 = nn.Conv2d(c5_layer, feature_dim, kernel_size=1, stride=1)
        self.p5_up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_2 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1)

        self.p4_1 = nn.Conv2d(c4_layer, feature_dim, kernel_size=1, stride=1)
        self.p4_up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_3 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1)

        self.p3_1 = nn.Conv2d(c3_layer, feature_dim, kernel_size=1, stride=1)
        self.p3_3 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1)

        self.p6 = nn.Conv2d(c5_layer, feature_dim, kernel_size=3, stride=2, padding=1)

        self.p7 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1))

    def forward(self, c3, c4, c5):
        # c3, c4, c5 = inputs

        p5_1 = self.p5_1(c5)
        p5_up = self.p5_up_sample(p5_1)
        p5_2 = self.p5_2(p5_1)

        p4_1 = self.p4_1(c4)
        p4_2 = p5_up + p4_1
        p4_up = self.p4_up_sample(p4_2)
        p4_3 = self.p4_3(p4_2)

        p3_1 = self.p3_1(c3)
        p3_2 = p3_1 + p4_up
        p3_3 = self.p3_3(p3_2)

        p3 = p3_3
        p4 = p4_3
        p5 = p5_2
        p6 = self.p6(c5)
        p7 = self.p7(p6)

        return [p3, p4, p5, p6, p7]


if __name__ == '__main__':
    fpn = FPN(256, 1024, 2048, 10)
    # x = []
    # x.append(torch.rand(1, 256, 76, 76))
    # x.append(torch.rand(1, 1024, 38, 38))
    # x.append(torch.rand(1, 2048, 19, 19))
    # c = fpn(x)


    summary(fpn, [(256, 76, 76), (1024, 38, 38), (2048, 19, 19)])


