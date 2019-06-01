import torch
import torch.functional
import torch.nn as nn

##############################
#           U-NET
##############################


class GeneratorUNetV1(nn.Module):
    def __init__(self, norm_layer='instance'):
        super(GeneratorUNetV1, self).__init__()

        ## ENCODE

        #self.norm = norm_layer
        
        self.d0 = FirstDownConvBlock(4,64, bias=False)
        self.d1 = DownBlockV2(64, 128, normalize=norm_layer, bias=False, dropout=0) # 64,128x128
        self.d2 = DownBlockV2(128, 256, normalize=norm_layer, bias=False, dropout=0) # 128,64x64
        self.d3 = DownBlockV2(256, 512, normalize=norm_layer, bias=False, dropout=0) # 256,32x32

        self.d4 = DownBlockV2(512, 512, normalize=norm_layer, bias=False, dropout=0.0) # 512,8x8
        self.d5 = DownBlockV2(512, 512, normalize=norm_layer, bias=False, dropout=0.0) # 512,4x4
        self.d6 = DownBlockV2(512, 512, normalize=norm_layer, bias=False, dropout=0.0) # 512,2x2
        
        self.d7 = DownBlockInner(512, 512, bias=False, dropout=0.0) # 512, 1x1

        ## DECODE

        self.u7 = UpBlockInner(512, 512, normalize=norm_layer, bias=False, dropout=0.0)    

        self.u6 = UpBlockV2(512, 512, normalize=norm_layer, bias=False, dropout=0.0)
        self.u5 = UpBlockV2(512, 512, normalize=norm_layer, bias=False, dropout=0.0)
        self.u4 = UpBlockV2(512, 512, normalize=norm_layer, bias=False, dropout=0.0)

        self.u3 = UpBlockV2(512, 256, normalize=norm_layer, bias=False, dropout=0.0)
        self.u2 = UpBlockV2(256, 128, normalize=norm_layer, bias=False, dropout=0.0)
        self.u1 = UpBlockV2(128, 64, normalize=norm_layer, bias=False, dropout=0.0)

        self.u0 = LastUpBlock(64, 3, bias=False, dropout=0.0)


        # self.inc = DoubleConvBlock(4, 32, normalize=norm_layer, bias=False, dropout=0)
        # self.d1 = DownBlock(32, 64, normalize=norm_layer, bias=False, dropout=0)
        # self.d2 = DownBlock(64, 128, normalize=norm_layer, bias=False, dropout=0.5)
        # self.d3 = DownBlock(128, 256, normalize=norm_layer, bias=False, dropout=0.5)
        # self.d4 = DownBlock(256, 512, normalize=norm_layer, bias=False, dropout=0.5)

        # self.u1 = UpBlock(512, 256, normalize=norm_layer, bias=False, dropout=0.5)
        # self.u2 = UpBlock(256, 128, normalize=norm_layer, bias=False, dropout=0.5)
        # self.u3 = UpBlock(128, 64, normalize=norm_layer, bias=False, dropout=0.5)
        # self.u4 = UpBlock(64, 32, normalize=norm_layer, bias=False, dropout=0)
        # self.outc = OutConvBLock(32, 12)
        # self.shuffle = nn.PixelShuffle(2)
        
    def forward(self, x):
        
        x0 = self.d0(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x5 = self.d5(x4)
        x6 = self.d6(x5)
        x7 = self.d7(x6)
        
        x = self.u7(x7, x6)
        x = self.u6(x, x5)
        x = self.u5(x, x4)
        x = self.u4(x, x3)
        x = self.u3(x, x2)
        x = self.u2(x, x1)
        x = self.u1(x, x0)
        x = self.u0(x)
        
        return x        
        # x1 = self.inc(x)
        # x2 = self.d1(x1)
        # x3 = self.d2(x2)
        # x4 = self.d3(x3)
        # x5 = self.d4(x4)
        # x = self.u1(x5, x4)
        # x = self.u2(x, x3)
        # x = self.u3(x, x2)
        # x = self.u4(x, x1)
        # x = self.outc(x)
        # return self.shuffle(x)

class Discriminator(nn.Module):
    def __init__(self, norm_layer='instance'):
        super().__init__()

        self.c1 = SingleConvBlock(6, 64, kernel_size=4, stride=2, bias=False)
        self.c2 = SingleConvBlock(64, 128, kernel_size=4, stride=2, bias=False, normalize=norm_layer)
        self.c3 = SingleConvBlock(128, 256, kernel_size=4, stride=2, bias=False, normalize=norm_layer)
        self.c4 = SingleConvBlock(256, 512, kernel_size=4, stride=2, bias=False, normalize=norm_layer)
        self.c5 = SingleConvBlock(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        #self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        #x = self.out(x)

        return x

class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchDiscriminator, self).__init__()

        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=True),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=True),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

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

class LastUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True, dropout=0):
        super(LastUpBlock, self).__init__()
        self.f = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(2 * in_ch, out_ch, kernel_size=4,
                                stride=2, padding=1, bias=bias),
                nn.Tanh()
                #nn.Sigmoid()
            )


    def forward(self, x):
        return self.f(x)

class FirstDownConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True, dropout=0):
        super(FirstDownConvBlock, self).__init__()
        model = [nn.Conv2d(in_ch, out_ch, kernel_size=4,
                               stride=2, padding=1, bias=bias)]

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

class UpBlockInner(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=None, bias=True, dropout=0):
        super(UpBlockInner, self).__init__()
        if normalize == 'batch':
            norm = nn.BatchNorm2d(out_ch)
        elif normalize == 'instance':
            norm = nn.InstanceNorm2d(out_ch)
        else:
            norm = None
        
        if norm:
            self.f = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                                stride=2, padding=1, bias=bias),
                norm
            )
        else:
            self.f = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                                stride=2, padding=1, bias=bias)
            )

    def forward(self, x1, x2):
        x = self.f(x1)
        return torch.cat([x2, x], dim=1)

class DownBlockInner(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True, dropout=0):
        super(DownBlockInner, self).__init__()
        self.f = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_ch, out_ch, kernel_size=4,
                                stride=2, padding=1, bias=bias),
            )

    def forward(self, x):
            return self.f(x)


class UpBlockV2(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=None, bias=True, dropout=0):
        super(UpBlockV2, self).__init__()
        if normalize == 'batch':
            norm = nn.BatchNorm2d(out_ch)
        elif normalize == 'instance':
            norm = nn.InstanceNorm2d(out_ch)
        else:
            norm = None
        
        if norm:
            self.f = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(2 * in_ch, out_ch, kernel_size=4,
                                stride=2, padding=1, bias=bias),
                norm
            )
        else:
            self.f = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(2 * in_ch, out_ch, kernel_size=4,
                                stride=2, padding=1, bias=bias)
            )
    def forward(self, x1, x2):
        x = self.f(x1)
        return torch.cat([x2, x], dim=1)



class DownBlockV2(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=None, bias=True, dropout=0):
        super(DownBlockV2, self).__init__()
        if normalize == 'batch':
            norm = nn.BatchNorm2d(out_ch)
        elif normalize == 'instance':
            norm = nn.InstanceNorm2d(out_ch)
        else:
            norm = None
        
        if norm:
            self.f = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_ch, out_ch, kernel_size=4,
                                stride=2, padding=1, bias=bias),
                norm
            )
        else:
            self.f = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_ch, out_ch, kernel_size=4,
                                stride=2, padding=1, bias=bias)
            )

    def forward(self, x):
        return self.f(x)

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
        
        self.f = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        #self.h = nn.Tanh()
        self.h = nn.Sigmoid()

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
