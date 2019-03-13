import torch
import torch.nn as nn


##############################
#   Initialize weights 
##############################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#   UNet Generator 
##############################

class GeneratorUNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(GeneratorUNet, self).__init__()


        model = [
            
        ]







class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.maxpool   = self.max_pool()

        self.conv1  = self.conv_3x3_2x(4, 32)        
        self.conv2  = self.conv_3x3_2x(32, 64)
        self.conv3  = self.conv_3x3_2x(64, 128)
        self.conv4  = self.conv_3x3_2x(128, 256)
        self.conv5  = self.conv_3x3_2x(256, 512)

        self.up6    = self.conv_up_2x2(512, 256)
        self.conv6  = self.conv_3x3_2x(512, 256)   
        self.up7    = self.conv_up_2x2(256, 128)
        self.conv7  = self.conv_3x3_2x(256, 128) 
        self.up8    = self.conv_up_2x2(128, 64)
        self.conv8  = self.conv_3x3_2x(128, 64)     
        self.up9    = self.conv_up_2x2(64, 32)
        self.conv9  = self.conv_3x3_2x(64, 32) 

        self.conv10 = self.conv_1x1(32,12)
    
    def forward(self, x):
        # Convolute input to 32 and max pool        
        c1  = self.conv1(x)
        m1  = self.maxpool(c1)

        # Convolute from 32 to 64 and max pool        
        c2  = self.conv2(m1)
        m2  = self.maxpool(c2)

        # Convolute from 64 to 128 and max pool        
        c3  = self.conv3(m2)
        m3  = self.maxpool(c3)

        # Convolute from 64 to 128 and max pool        
        c4  = self.conv4(m3)
        m4  = self.maxpool(c4)

        c5  = self.conv5(m4)

        up6 = self.copy_and_crop(self.up6(c5), c4)
        c6  = self.conv6(up6)

        up7 = self.copy_and_crop(self.up7(c6), c3)
        c7  = self.conv7(up7)

        up8 = self.copy_and_crop(self.up8(c7), c2)
        c8  = self.conv8(up8)

        up9 = self.copy_and_crop(self.up9(c8), c1)
        c9  = self.conv9(up9)

        c10 = self.conv10(c9)
        
        return nn.functional.pixel_shuffle(c10, 2)

    def conv_up_2x2(self, in_channels, out_channels):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,stride=2))

    def conv_1x1(self, in_channels, out_channels, batch_norm=False):
        
        sequence = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)]

        if batch_norm:
            sequence.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*sequence)

    def conv_3x3_2x(self, in_channels, out_channels, batch_norm=False):
        # 3x3 convolutional layer with stide 1 and padding 1
        
        # That is, when F=3, then using P=1 will retain the original size of the input.
        sequence = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)]
        
        if batch_norm:
            sequence.append(nn.BatchNorm2d(out_channels))

        sequence.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        sequence.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))

        if batch_norm:
            sequence.append(nn.BatchNorm2d(out_channels))

        sequence.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        return nn.Sequential(*sequence)

    def max_pool(self):
        # The most common setting is to use max-pooling with 2x2 receptive fields (i.e. F=2), and with a stride of 2 (i.e. S=2).
        return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

    def copy_and_crop(self, expansive_layer, contracting_layer):
        return torch.cat([expansive_layer, contracting_layer], 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    
