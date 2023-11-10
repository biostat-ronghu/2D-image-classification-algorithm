import torch
# import torchvision
# import numpy as np
import time
from thop import profile
from torch import nn
from torch.nn import functional as F


class Residual_1(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual_1(input_channels, num_channels,
                                  use_1x1conv=True, strides=2))
        else:
            blk.append(Residual_1(num_channels, num_channels))
    return blk


class ResNet_18or34(nn.Module):
    def __init__(self, num_residuals_b2, num_residuals_b3, num_residuals_b4, num_residuals_b5, class_cnt):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(
            *resnet_block(64, 64, num_residuals_b2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, num_residuals_b3))
        self.b4 = nn.Sequential(*resnet_block(128, 256, num_residuals_b4))
        self.b5 = nn.Sequential(*resnet_block(256, 512, num_residuals_b5))
        self.resnet = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5,
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten(), nn.Linear(512, class_cnt))

    def forward(self, X):
        Y = self.resnet(X)
        return Y


def cal_fps(model):
    time_1 = time.time()
    for _ in range(50):
        input_image = torch.randn(1,3,224,224)
        pred = model(input_image)

    time_2 = time.time()
    fps = 50/(time_2-time_1)
    return fps



resnet_final = ResNet_18or34(3,4,6,3,10)
resnet_final.load_state_dict(torch.load(r"D:\work_files\task\model_weights\best_resnet_weights-resnet36.pth"))
print( 'The fps of resnet_final is ',cal_fps(resnet_final) )
flops, params = profile(resnet_final, (torch.randn(1, 3, 224, 224),))
print(f'flops of resnet_final: {flops/1e6:>.2f} M, params of resnet_final: {params/1e6:>.2f} M')

print('--------------------------------------------------')


vit_final = torch.load(r"D:\work_files\task\model_weights\best_vit.pth")
print( 'The fps of vit_final is ',cal_fps(vit_final) )
flops, params = profile(vit_final, (torch.randn(1, 3, 224, 224),))
print(f'flops of vit_final: {flops/1e6:>.2f} M, params of vit_final: {params/1e6:>.2f} M')

print('--------------------------------------------------')


swin_final = torch.load(r"D:\work_files\task\model_weights\best_swin2.pth")
print( 'The fps of swin_final is ',cal_fps(swin_final) )
flops, params = profile(swin_final, (torch.randn(1, 3, 224, 224),))
print(f'flops of swin_final: {flops/1e6:>.2f} M, params of swin_final: {params/1e6:>.2f} M')