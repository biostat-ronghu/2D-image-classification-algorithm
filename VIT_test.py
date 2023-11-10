import os
import pandas as pd

import random
import shutil
random.seed(-1)

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Lambda,ToTensor

from torch.utils.data import DataLoader
import numpy as np
import time

test_path = r"D:\work_files\task\test_data"

transform_3 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.51805955, 0.5003706, 0.4121268), std=(0.21985851, 0.22019151, 0.2216588))
])

test_data = torchvision.datasets.ImageFolder(test_path,transform=transform_3)

batch_size = 64
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

start_time = time.time()

vit_final = torch.load(r"D:\work_files\task\model_weights\best_vit.pth")
vit_final.eval()

class_cnt = 10

dataloader = test_dataloader
size = len(dataloader.dataset)
num_batches = len(dataloader)
correct = 0
confusion = np.zeros((class_cnt, class_cnt))

with torch.no_grad():
    for X, y in dataloader:
        pred = vit_final(X)
        y_pred = pred.argmax(1)
        correct += (y_pred == y).type(torch.float).sum().item()

        for i in range(len(y)):
            confusion[[y_pred[i]], [y[i]]] += 1

correct /= size

print(f'Test took {time.time()-start_time} seconds!')

precision_list = [confusion[i, i] / sum(confusion[i, :]) if sum(confusion[i, :]) != 0 else 0 for i in range(class_cnt)]
recall_list = [confusion[i, i] / sum(confusion[:, i]) if sum(confusion[:, i]) != 0 else 0 for i in range(class_cnt)]
f1_score = np.mean([2 * (p * r) / (p + r) if p + r != 0 else 0 for p, r in zip(precision_list, recall_list)])
print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, F1-score: {(100 * f1_score):>0.1f}% \n")