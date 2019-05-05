import torch
import torch.functional
import torch.nn as nn

##############################
#           U-NET
##############################


class GeneratorUNetV1(nn.Module):
    def __init__(self, norm_layer='instance'):
        super(GeneratorUNetV1, self).__init__()

        self.norm = norm_layer
        self.inc = DoubleConvBlock(3, 32, normalize=norm_layer, bias=True, dropout=0.0)
        self.d1 = DownBlock(32, 64, normalize=norm_layer, bias=True, dropout=0)
        self.d2 = DownBlock(64, 128, normalize=norm_layer, bias=True, dropout=0.5)
        self.d3 = DownBlock(128, 256, normalize=norm_layer, bias=True, dropout=0.5)
        self.d4 = DownBlock(256, 512, normalize=norm_layer, bias=True, dropout=0.5)

        self.u1 = UpBlock(512, 256, normalize=norm_layer, bias=True, dropout=0.5)
        self.u2 = UpBlock(256, 128, normalize=norm_layer, bias=True, dropout=0.5)
        self.u3 = UpBlock(128, 64, normalize=norm_layer, bias=True, dropout=0)
        self.u4 = UpBlock(64, 32, normalize=norm_layer, bias=True, dropout=0)
        self.outc = OutConvBLock(32, 3)
        
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
        return x


class Discriminator(nn.Module):
    def __init__(self, norm_layer='instance'):
        super().__init__()

        self.c1 = SingleConvBlock(6, 64, kernel_size=4, stride=2, bias=False)
        self.c2 = SingleConvBlock(64, 128, kernel_size=4, stride=2, bias=False, normalize=norm_layer)
        self.c3 = SingleConvBlock(128, 256, kernel_size=4, stride=2, bias=False, normalize=norm_layer)
        self.c4 = SingleConvBlock(256, 512, kernel_size=4, stride=2, bias=False, normalize=norm_layer)
        self.c5 = SingleConvBlock(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.out(x)

        return x

class DiscriminatorPatch(nn.Module):
    def __init__(self, num_layers=3, norm_layer=None):
        super().__init__()

        model = []

        for i in range(1, num_layers):
            pass

    def forward(self, x):

        return x


class SingleConvBlock(nn.Module):
    def __init__(
        self,
        in_ch, 
        out_ch, 
        normalize=None, 
        bias=True, 
        dropout=0, 
        kernel_size=3, 
        stride=1, 
        padding=1
     ):
        super(SingleConvBlock, self).__init__()
        
        model = [nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                   stride=1, padding=1, bias=bias)]
        
        if normalize == 'batch':
                model.append(nn.BatchNorm2d(out_ch))
        elif normalize == 'instance':
            model.append(nn.InstanceNorm2d(out_ch))

        model.append(nn.LeakyReLU(0,2))

        if dropout > 0:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=None, bias=True, dropout=0):
        super(DoubleConvBlock, self).__init__()

        if normalize == 'batch':
            model = [nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=bias)]
            model.append(nn.BatchNorm2d(out_ch))
            model.append(nn.LeakyReLU(0,2))
            model.append(nn.Conv2d(out_ch, out_ch, kernel_size=3,
                                   stride=1, padding=1, bias=bias))
            model.append(nn.BatchNorm2d(out_ch))
            model.append(nn.LeakyReLU(0,2))

        elif normalize == 'instance':
            model = [nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=bias)]
            model.append(nn.InstanceNorm2d(out_ch))
            model.append(nn.LeakyReLU(0,2))
            model.append(nn.Conv2d(out_ch, out_ch, kernel_size=3,
                                   stride=1, padding=1, bias=bias))
            model.append(nn.InstanceNorm2d(out_ch))
            model.append(nn.LeakyReLU(0,2))
        else:
            model = [nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=bias)]
            model.append(nn.LeakyReLU(0,2))
            model.append(nn.Conv2d(out_ch, out_ch, kernel_size=3,
                                   stride=1, padding=1, bias=bias))
            model.append(nn.LeakyReLU(0,2))

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
        self.f = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                           stride=1, padding=1, bias=True)
        self.h = nn.Tanh()

    def forward(self, x):
        x = self.f(x)
        x = self.h(x)
        return x


class GAN_loss(nn.Module):
    """ Compute GAN loss """

    def __init__(self, loss, device):
        super().__init__()

        if loss == 'BCE':
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

        self.targets = {
            0.0: torch.tensor(0.0).to(device),
            1.0: torch.tensor(1.0).to(device)
        }

    def create_target(self, size, target):
        # Create target tensor of size
        t_target = self.targets[target]
        t_target = t_target.expand(size)
        return t_target

    def compute(self, t_prediction, target):
        # Size of prediction tensor
        size = t_prediction.size()
        # Get target tensor of same size
        t_target = self.create_target(size, target)
        # Compute loss
        loss = self.loss(t_prediction, t_target)
        return loss
