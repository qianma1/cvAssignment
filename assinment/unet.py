import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):  # 双层卷积模型，神经网络最基本的框架
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),  # 加入Bn层提高网络泛化能力（防止过拟合），加收敛速度
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),  # 3指kernel_size，即卷积核3*3
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# class UpSample(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
#         super(UpSample, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1)
#         self.conv_relu = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.BatchNorm2d(num_features=out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.BatchNorm2d(num_features=out_channels),
#             nn.ReLU(),
#         )
#
#     def forward(self, x, y):
#         x = self.up(x)
#         x1 = torch.cat((x, y), dim=0)
#         x1 = self.conv_relu(x1)
#         return x1 + x


class AdUNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # 改为512*512的初始图像，将下面的数据预处理过程消减
        self.adnet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 1), padding=0, stride=(2, 1)),
            nn.BatchNorm2d(1),  # 加入Bn层提高网络泛化能力（防止过拟合），加收敛速度
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=(1, 2), padding=0, stride=(1, 2)),
            nn.BatchNorm2d(1),  # 加入Bn层提高网络泛化能力（防止过拟合），加收敛速度
            nn.ReLU(inplace=True)
        )
        
        self.dconv_down0 = double_conv(1, 32)
        self.dconv_down1 = double_conv(32, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        
        self.maxpool = nn.MaxPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.upsample1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.upsample0 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        
        self.dconv_up3 = double_conv(256 + 256, 256)
        self.dconv_up2 = double_conv(128 + 128, 128)
        self.dconv_up1 = double_conv(64 + 64, 64)
        self.dconv_up0 = double_conv(64, 32)
        
        self.conv_last = nn.Conv2d(16, 1, 1)
    
    def forward(self, x):
        # reshape
        x = self.adnet(x)  # 1x128x128   #注释掉，回归最原始的模型
        
        # encode
        conv0 = self.dconv_down0(x)  # 32x128x128
        x = self.maxpool(conv0)  # 32x64x64
        
        conv1 = self.dconv_down1(x)  # 64x64x64
        x = self.maxpool(conv1)  # 64x32x32
        
        conv2 = self.dconv_down2(x)  # 128x32x32
        x = self.maxpool(conv2)  # 128x16x16
        
        conv3 = self.dconv_down3(x)  # 256x16x16
        x = self.maxpool(conv3)  # 256x8x8
        
        x = self.dconv_down4(x)  # 512x8x8
        # x = self.maxpool(conv3)  # 512x4x4 # 手动新增
        
        # decode
        x = self.upsample4(x)  # 256x16x16
        # 因为使用了3*3卷积核和 padding=1 的组合，所以卷积过程图像尺寸不发生改变，所以省去了crop操作！
        x = torch.cat([x, conv3], dim=1)  # 512x16x16
        
        x = self.dconv_up3(x)  # 256x16x16
        x = self.upsample3(x)  # 128x32x32
        x = torch.cat([x, conv2], dim=1)  # 256x32x32
        
        x = self.dconv_up2(x)  # 128x32x32
        x = self.upsample2(x)  # 64x64x64
        x = torch.cat([x, conv1], dim=1)  # 128x64x64
        
        x = self.dconv_up1(x)  # 64x64x64
        x = self.upsample1(x)  # 32x128x128
        x = torch.cat([x, conv0], dim=1)  # 64x128x128
        
        x = self.dconv_up0(x)  # 32x128x128
        x = self.upsample0(x)  # 16x256x256
        
        out = self.conv_last(x)  # 1x256x256
        
        return out
