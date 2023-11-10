import random
random.seed(-1)

import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader
import numpy as np


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
            *resnet_block(64, 64, num_residuals_b2, first_block=True))  # resnet_block的返回值类型为list,因此需要用*
        self.b3 = nn.Sequential(*resnet_block(64, 128, num_residuals_b3))
        self.b4 = nn.Sequential(*resnet_block(128, 256, num_residuals_b4))
        self.b5 = nn.Sequential(*resnet_block(256, 512, num_residuals_b5))
        self.resnet = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5,
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten(), nn.Linear(512, class_cnt))

    def forward(self, X):
        Y = self.resnet(X)
        return Y



test_path = r"D:\work_files\task\test_data"

transform_3 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.51805955, 0.5003706, 0.4121268), std=(0.21985851, 0.22019151, 0.2216588))
])

test_data = torchvision.datasets.ImageFolder(test_path,transform=transform_3)

batch_size = 64
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

class_cnt = 10

model_final = ResNet_18or34(3, 4, 6, 3, class_cnt)
model_final.load_state_dict(torch.load(r"D:\work_files\task\model_weights\best_resnet_weights-resnet36.pth"))
model_final.eval()

dataloader = test_dataloader
size = len(dataloader.dataset)
num_batches = len(dataloader)
correct = 0
confusion = np.zeros((class_cnt, class_cnt))

with torch.no_grad():
    for X, y in dataloader:
        pred = model_final(X)
        y_pred = pred.argmax(1)
        correct += (y_pred == y).type(torch.float).sum().item()

        for i in range(len(y)):
            confusion[[y_pred[i]], [y[i]]] += 1

correct /= size

precision_list = [confusion[i, i] / sum(confusion[i, :]) if sum(confusion[i, :]) != 0 else 0 for i in range(class_cnt)]
recall_list = [confusion[i, i] / sum(confusion[:, i]) if sum(confusion[:, i]) != 0 else 0 for i in range(class_cnt)]
f1_score = np.mean([2 * (p * r) / (p + r) if p + r != 0 else 0 for p, r in zip(precision_list, recall_list)])
print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, F1-score: {(100 * f1_score):>0.1f}% \n")