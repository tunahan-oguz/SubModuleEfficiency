from __future__ import annotations

import torch
import torch.nn as nn

from train_app.models import base
from train_app.registers.model_registry import model_registry

class DoubleConv(nn.Module):
    def __init__(self, in_chann, out_chann, mid_chann=None):
        super().__init__()

        if mid_chann is None:
            mid_chann = out_chann

        self.doubleConv = nn.Sequential(
            nn.Conv2d(in_channels=in_chann, out_channels=mid_chann, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_chann),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_chann, out_channels=out_chann, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_chann),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.doubleConv(x)


class Down(nn.Module):
    def __init__(self, in_chann, out_chann, mid_chann=None):
        super().__init__()

        if mid_chann is None:
            mid_chann = out_chann

        self.doubleConv = DoubleConv(in_chann=in_chann, mid_chann=mid_chann, out_chann=out_chann)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxPool(x)
        x = self.doubleConv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_chann, out_chann):
        super().__init__()

        self.deConv = nn.ConvTranspose2d(
            in_channels=in_chann, out_channels=in_chann // 2, kernel_size=2, stride=2
        )  # Decrease the channel depth to half
        self.doubleConv = DoubleConv(in_chann=in_chann, out_chann=out_chann)  # After concat channel depth will be eq to in_chann

    def forward(self, x1, x2):
        x1 = self.deConv(x1)
        x2 = torch.cat((x1, x2), dim=1)

        return self.doubleConv(x2)

@model_registry.register("UNet")
class UNet(base.SemanticSegmentationAdapter):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inc = DoubleConv(in_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        inc = self.inc(x)
        d1 = self.down1(inc)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        x = self.up1(d4, d3)
        x = self.up2(x, d2)
        x = self.up3(x, d1)
        x = self.up4(x, inc)

        return self.conv(x)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
    
    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

@model_registry.register("AttU_Net")
class AttU_Net(base.SemanticSegmentationAdapter):
    def __init__(self,img_ch=3,output_ch=1, *args, **kwargs):
        super(AttU_Net,self).__init__(*args, **kwargs)
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1