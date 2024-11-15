import torch
import torch.nn as nn
from .complexLayers import *

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine = True)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels, affine = True),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat((x,skip),1)
        return x

# generator: 가짜 이미지를 생성합니다.
class Unet_Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 128 x 128 x 1
        self.down1 = UNetDown(in_channels, 128, normalize=False)
        # 64 x 64 x 128
        self.down2 = UNetDown(128,256)
        # 32 x 32 x 256
        self.down3 = UNetDown(256,512)
        # 16 x 16 x 512
        self.down4 = UNetDown(512,512)
        # 8 x 8 x 512
        self.down5 = UNetDown(512,512)
        # 4 x 4 x 512
        self.down6 = UNetDown(512,512)
        # 2 x 2 x 512
        self.down7 = UNetDown(512,512,normalize=False)
        # 1 x 1 x 512

        self.up1 = UNetUp(512,512,dropout=0.5)
        # 2 x 2 x 1024
        self.up2 = UNetUp(1024,512,dropout=0.5)
        # 4 x 4 x 1024
        self.up3 = UNetUp(1024,512,dropout=0.5)
        # 8 x 8 x 1024
        self.up4 = UNetUp(1024,512)
        # 16 x 16 x 1024
        self.up5 = UNetUp(1024,256)
        # 32 x 32 x 512
        self.up6 = UNetUp(512,128)
        # 64 x 64 x 256
        self.up7 = nn.ConvTranspose2d(256,out_channels,4,stride=2,padding=1)
        self.tanh = nn.Tanh()
        # 128 x 128 x 1

    def forward(self, x):
        d1 = self.down1(x) # 128
        d2 = self.down2(d1) # 256
        d3 = self.down3(d2) # 512
        d4 = self.down4(d3) # 512
        d5 = self.down5(d4) # 512
        d6 = self.down6(d5) # 512
        d7 = self.down7(d6) # 512

        u1 = self.up1(d7,d6) # input : 512, 512, output : 1024
        u2 = self.up2(u1,d5) # input : 1024, 512, output : 1024
        u3 = self.up3(u2,d4) # input : 1024, 512, output : 1024
        u4 = self.up4(u3,d3) # input : 1024, 512, output : 1024
        u5 = self.up5(u4,d2) # input : 1024, 256, output : 512
        u6 = self.up6(u5,d1) # input : 512, 128, output : 256
        u7 = self.up7(u6)    # input : 256,      output : 1
        out = self.tanh(u7)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.conv_block = nn.Sequential(nn.ReflectionPad2d(1),
                                        nn.Conv2d(in_channel, in_channel, 3),
                                        nn.InstanceNorm2d(in_channel),
                                        nn.ReLU(inplace=True),
                                        nn.ReflectionPad2d(1),
                                        nn.Conv2d(in_channel, in_channel, 3),
                                        nn.InstanceNorm2d(in_channel))
        
    def forward(self, x):
        return x + self.conv_block(x)
    
class Resnet_Generator(nn.Module):
    def __init__(self, in_channel = 1, out_channel = 1, n_residual_block = 6, bias = True):
        super().__init__()

        conv_dim = 64

        # Down Sampling Layer
        # 128 x 128 x 1
        down_layers = []
        down_layers.append(nn.ReflectionPad2d(3))
        down_layers.append(nn.Conv2d(in_channel, conv_dim, kernel_size=7, bias=bias))
        down_layers.append(nn.InstanceNorm2d(conv_dim))
        down_layers.append(nn.ReLU(inplace=True))
        
        # 128 x 128 x 64
        down_layers.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1, bias=bias))
        down_layers.append(nn.InstanceNorm2d(conv_dim*2))
        down_layers.append(nn.ReLU(inplace=True))

        # 64 x 64 x 128
        down_layers.append(nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1, bias=bias))
        down_layers.append(nn.InstanceNorm2d(conv_dim*4))
        down_layers.append(nn.ReLU(inplace=True))

        # Bottleneck Layer
        # 32 x 32 x 256
        bottle_layer = []
        for i in range(n_residual_block):
            bottle_layer.append(ResidualBlock(conv_dim*4))

        # Up Sampling Layer
        # 32 x 32 x 256
        up_layer = []
        up_layer.append(nn.ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias))
        up_layer.append(nn.InstanceNorm2d(conv_dim*2))
        up_layer.append(nn.ReLU(inplace=True))

        # 64 x 64 x 128
        up_layer.append(nn.ConvTranspose2d(conv_dim*2, conv_dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias))
        up_layer.append(nn.InstanceNorm2d(conv_dim))
        up_layer.append(nn.ReLU(inplace=True))

        # 128 x 128 x 64
        up_layer.append(nn.ReflectionPad2d(3))
        up_layer.append(nn.Conv2d(conv_dim, out_channel, kernel_size=7, bias=bias))
        up_layer.append(nn.Tanh())

        # 128 x 128 x 1
        self.down = nn.Sequential(*down_layers)
        self.bottle = nn.Sequential(*bottle_layer)
        self.up = nn.Sequential(*up_layer)

    def forward(self, x):
        x = self.down(x)
        x = self.bottle(x)
        x = self.up(x)

        return x


# Complex UNet
class ComplexUNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [ComplexConv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(ComplexInstanceNorm2d(out_channels, affine = True)),

        layers.append(ComplexLeakyReLU(0.2))

        if dropout:
            layers.append(ComplexDropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x

class ComplexUNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            ComplexConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            ComplexInstanceNorm2d(out_channels, affine = True),
            ComplexLeakyReLU()
        ]

        if dropout:
            layers.append(ComplexDropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat((x,skip),1)
        return x

# generator: 가짜 이미지를 생성합니다.
class ComplexGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 128 x 128 x 1
        self.down1 = ComplexUNetDown(in_channels, 128, normalize=False)
        # 64 x 64 x 128
        self.down2 = ComplexUNetDown(128,256)
        # 32 x 32 x 256
        self.down3 = ComplexUNetDown(256,512)
        # 16 x 16 x 512
        self.down4 = ComplexUNetDown(512,512)
        # 8 x 8 x 512
        self.down5 = ComplexUNetDown(512,512)
        # 4 x 4 x 512
        self.down6 = ComplexUNetDown(512,512)
        # 2 x 2 x 512
        self.down7 = ComplexUNetDown(512,512,normalize=False)
        # 1 x 1 x 512

        self.up1 = ComplexUNetUp(512,512,dropout=0.5)
        # 2 x 2 x 1024
        self.up2 = ComplexUNetUp(1024,512,dropout=0.5)
        # 4 x 4 x 1024
        self.up3 = ComplexUNetUp(1024,512,dropout=0.5)
        # 8 x 8 x 1024
        self.up4 = ComplexUNetUp(1024,512)
        # 16 x 16 x 1024
        self.up5 = ComplexUNetUp(1024,256)
        # 32 x 32 x 512
        self.up6 = ComplexUNetUp(512,128)
        # 64 x 64 x 256
        self.up7 = ComplexConvTranspose2d(256,out_channels,4,stride=2,padding=1)
        self.tanh = nn.Tanh()
        # 128 x 128 x 1

    def forward(self, x):
        d1 = self.down1(x) # 128
        d2 = self.down2(d1) # 256
        d3 = self.down3(d2) # 512
        d4 = self.down4(d3) # 512
        d5 = self.down5(d4) # 512
        d6 = self.down6(d5) # 512
        d7 = self.down7(d6) # 512

        u1 = self.up1(d7,d6) # input : 512, 512, output : 1024
        u2 = self.up2(u1,d5) # input : 1024, 512, output : 1024
        u3 = self.up3(u2,d4) # input : 1024, 512, output : 1024
        u4 = self.up4(u3,d3) # input : 1024, 512, output : 1024
        u5 = self.up5(u4,d2) # input : 1024, 256, output : 512
        u6 = self.up6(u5,d1) # input : 512, 128, output : 256
        u7 = abs(self.up7(u6))    # input : 256,      output : 1
        out = self.tanh(u7)

        return out

class Complex_Residual_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Complex_Residual_Block, self).__init__()
        self.main = nn.Sequential(
            ComplexConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            ComplexInstanceNorm2d(out_channel, affine=True),
            ComplexReLU(),
            ComplexConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            ComplexInstanceNorm2d(out_channel)
        )
    def forward(self, x):
        return x + self.main(x)
    
class ComplexGenerator2(nn.Module):
    def __init__(self, in_channel = 1, out_channel = 1, repeat_num = 6):
        super(ComplexGenerator2, self).__init__()

        conv_dim = 64

        # Down Sampling Layer
        down_layers = []
        down_layers.append(ComplexConv2d(in_channel, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        down_layers.append(ComplexInstanceNorm2d(conv_dim, affine=True))
        down_layers.append(ComplexReLU())

        down_layers.append(ComplexConv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        down_layers.append(ComplexInstanceNorm2d(conv_dim*2, affine=True))
        down_layers.append(ComplexReLU())

        down_layers.append(ComplexConv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1, bias=False))
        down_layers.append(ComplexInstanceNorm2d(conv_dim*4, affine=True))
        down_layers.append(ComplexReLU())

        # Bottleneck Layer
        bottle_layers = []
        for i in range(repeat_num):
            bottle_layers.append(Complex_Residual_Block(conv_dim*4, conv_dim*4))

        # Up Sampling Layer
        up_layers = []
        up_layers.append(ComplexConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        up_layers.append(ComplexInstanceNorm2d(conv_dim*2, affine=True))
        up_layers.append(ComplexReLU())

        up_layers.append(ComplexConvTranspose2d(conv_dim*2, conv_dim, kernel_size=4, stride=2, padding=1, bias=False))
        up_layers.append(ComplexInstanceNorm2d(conv_dim, affine=True))
        up_layers.append(ComplexReLU())

        up_layers.append(ComplexConv2d(conv_dim, out_channel, kernel_size=7, stride=1, padding=3, bias=False))
        
        self.down = nn.Sequential(*down_layers)
        self.bottle = nn.Sequential(*bottle_layers)
        self.up = nn.Sequential(*up_layers)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.down(x)
        x = self.bottle(x)
        x = abs(self.up(x))
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channel=2):
        super().__init__()

        conv_dim = 64

        # 128 x 128 x 1
        model = []
        model.append(nn.Conv2d(in_channel, conv_dim, kernel_size=4, stride=2, padding=1))
        model.append(nn.LeakyReLU(0.2, inplace=True))

        # 64 x 64 x 64
        model.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1))
        model.append(nn.InstanceNorm2d(conv_dim*2))
        model.append(nn.LeakyReLU(0.2, inplace=True))

        # 32 x 32 x 128
        model.append(nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1))
        model.append(nn.InstanceNorm2d(conv_dim*4))
        model.append(nn.LeakyReLU(0.2, inplace=True))

        # 16 x 16 x 256
        model.append(nn.Conv2d(conv_dim*4, 1, kernel_size=4, padding=1))

        # 15 x 15 x 1
        self.model = nn.Sequential(*model)

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        x = self.model(x)
        x = torch.sigmoid(x)
        return x