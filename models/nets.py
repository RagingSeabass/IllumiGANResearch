import torch
import torch.functional
import torch.nn as nn

##############################
#           U-NET
##############################


class GeneratorUNetV1(nn.Module):
    def __init__(self, norm_layer='instance', use_dropout=False):
        super(GeneratorUNetV1, self).__init__()

        # TODO: Look at dropout implementation

        self.norm = norm_layer
        self.inc = DoubleConvBlock(4, 32, normalize=norm_layer)
        self.d1 = DownBlock(32, 64, normalize=norm_layer)
        self.d2 = DownBlock(64, 128, normalize=norm_layer)
        self.d3 = DownBlock(128, 256, normalize=norm_layer)
        self.d4 = DownBlock(256, 512, normalize=norm_layer)

        self.u1 = UpBlock(512, 256, normalize=norm_layer)
        self.u2 = UpBlock(256, 128, normalize=norm_layer)
        self.u3 = UpBlock(128, 64, normalize=norm_layer)
        self.u4 = UpBlock(64, 32, normalize=norm_layer)
        self.outc = OutConvBLock(32, 12)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        x = self.outc(x)
        x = nn.functional.pixel_shuffle(x, 2)

        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=None, bias=True, dropout=0):
        super(DoubleConvBlock, self).__init__()

        if normalize == 'batch':
            model = [nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=bias)]
            model.append(nn.BatchNorm2d(out_ch))
            model.append(nn.LeakyReLU(0.2))
            model.append(nn.Conv2d(out_ch, out_ch, kernel_size=3,
                                   stride=1, padding=1, bias=bias))
            model.append(nn.BatchNorm2d(out_ch))
            model.append(nn.LeakyReLU(0.2))

        elif normalize == 'instance':
            model = [nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=bias)]
            model.append(nn.InstanceNorm2d(out_ch))
            model.append(nn.LeakyReLU(0.2))
            model.append(nn.Conv2d(out_ch, out_ch, kernel_size=3,
                                   stride=1, padding=1, bias=bias))
            model.append(nn.InstanceNorm2d(out_ch))
            model.append(nn.LeakyReLU(0.2))
        else:
            model = [nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=bias)]
            model.append(nn.LeakyReLU(0.2))
            model.append(nn.Conv2d(out_ch, out_ch, kernel_size=3,
                                   stride=1, padding=1, bias=bias))
            model.append(nn.LeakyReLU(0.2))

        if dropout > 0:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=None, bias=True, dropout=0):
        super(DownBlock, self).__init__()
        self.f = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_ch, out_ch, normalize=normalize,
                            bias=bias, dropout=dropout),
        )

    def forward(self, x):
        x = self.f(x)
        return x


class UpBlock(nn.Module):
    # upsample and concat

    def __init__(self, in_ch, out_ch, normalize=None, bias=True, dropout=0):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias)
        self.conv = DoubleConvBlock(
            in_ch, out_ch, normalize=normalize, bias=bias, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConvBLock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConvBLock, self).__init__()
        self.f = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                           stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.f(x)
        return x
