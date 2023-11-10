import os
import pandas as pd
from torch.utils.data import DataLoader
import random
import shutil
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Lambda,ToTensor
from torch import nn
from torch.nn import functional as F
import time
import copy
import numpy as np

folder_path_all = r"D:\work_files\task\Animals-10\raw-img"
csv_filename_all = r"D:\work_files\task\Animals10_labels.csv"  # .csv文件名

image_files_all = []
labels_all = []
for i in range(len(os.listdir(folder_path_all))):
    folder_path = folder_path_all + '\\' + os.listdir(folder_path_all)[i]
    image_files = [filename for filename in os.listdir(folder_path)
                   if (filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"))]
    if len(image_files) != len(os.listdir(folder_path)):
        print('Take care of the format of each image file!')

    image_files_all += image_files
    labels_all += [i] * len(image_files)  # 类别编号从0开始
    print(f'The number of {os.listdir(folder_path_all)[i]} is:', len(image_files))

result = pd.DataFrame({'filename': image_files_all, 'label': labels_all})

print('图片总数为:',len(result))


random.seed(-1)


# test_paitition = 0.2

# for i in range(len(os.listdir(folder_path_all))):

#     os.mkdir( r'D:\work_files\task\train_val_data' + '\\' + os.listdir(folder_path_all)[i] )
#     os.mkdir( r'D:\work_files\task\test_data' + '\\' + os.listdir(folder_path_all)[i] )

#     folder_path = folder_path_all + '\\' + os.listdir(folder_path_all)[i]
#     image_idx = list(range(len(os.listdir(folder_path))))
#     random.shuffle(image_idx)
#     image_idx = image_idx[0:round(test_paitition*len(os.listdir(folder_path)))]

#     for j in range( len(os.listdir(folder_path)) ):
#         if j in image_idx:
#             shutil.copy( os.path.join( os.path.join( folder_path_all,os.listdir(folder_path_all)[i] ) ,os.listdir(os.path.join( folder_path_all,os.listdir(folder_path_all)[i] ))[j] ) ,
#                         r'D:\work_files\task\test_data' + '\\' + os.listdir(folder_path_all)[i])
#         else:
#             shutil.copy( os.path.join( os.path.join( folder_path_all,os.listdir(folder_path_all)[i] ) ,os.listdir(os.path.join( folder_path_all,os.listdir(folder_path_all)[i] ))[j] ) ,
#                         r'D:\work_files\task\train_val_data' + '\\' + os.listdir(folder_path_all)[i])


# folder_path_all = r"D:\work_files\task\train_val_data"
# val_paitition = 0.2

# for i in range(len(os.listdir(folder_path_all))):

#     os.mkdir( r'D:\work_files\task\train_data' + '\\' + os.listdir(folder_path_all)[i] )
#     os.mkdir( r'D:\work_files\task\val_data' + '\\' + os.listdir(folder_path_all)[i] )

#     folder_path = folder_path_all + '\\' + os.listdir(folder_path_all)[i]
#     image_idx = list(range(len(os.listdir(folder_path))))
#     random.shuffle(image_idx)
#     image_idx = image_idx[0:round(test_paitition*len(os.listdir(folder_path)))]

#     for j in range( len(os.listdir(folder_path)) ):
#         if j in image_idx:
#             shutil.copy( os.path.join( os.path.join( folder_path_all,os.listdir(folder_path_all)[i] ) ,os.listdir(os.path.join( folder_path_all,os.listdir(folder_path_all)[i] ))[j] ) ,
#                         r'D:\work_files\task\val_data' + '\\' + os.listdir(folder_path_all)[i])
#         else:
#             shutil.copy( os.path.join( os.path.join( folder_path_all,os.listdir(folder_path_all)[i] ) ,os.listdir(os.path.join( folder_path_all,os.listdir(folder_path_all)[i] ))[j] ) ,
#                         r'D:\work_files\task\train_data' + '\\' + os.listdir(folder_path_all)[i])




train_path = r"D:\work_files\task\train_data"
val_path = r"D:\work_files\task\val_data"
test_path = r"D:\work_files\task\test_data"

# def getStat(my_data):
#     '''
#     Compute mean and variance for data
#     :param my_data: 自定义类Dataset(或ImageFolder即可)
#     :return: (mean, std)
#     '''
#     print('The sample size of data is:',len(my_data))
#     data_loader = torch.utils.data.DataLoader(
#         my_data, batch_size=1, shuffle=False, num_workers=0,
#         pin_memory=True)
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     for X, _ in data_loader:
#         for d in range(1):
#             mean[d] += X[:, d, :, :].mean()
#             std[d] += X[:, d, :, :].std()
#     mean.div_(len(my_data))
#     std.div_(len(my_data))
#     return list(mean.numpy()), list(std.numpy())

# my_data_train = torchvision.datasets.ImageFolder(root=train_path,
#                                                 transform=transforms.Compose([transforms.Grayscale(1),transforms.ToTensor()]))
# print('Compute mean and variance for training data.\n',getStat(my_data_train))

# my_data_val = torchvision.datasets.ImageFolder(root=val_path,
#                                               transform=transforms.Compose([transforms.Grayscale(1),transforms.ToTensor()]))
# print('Compute mean and variance for val data.\n',getStat(my_data_val))

# my_data_test = torchvision.datasets.ImageFolder(root=test_path,
#                                                transform=transforms.Compose([transforms.Grayscale(1),transforms.ToTensor()]))
# print('Compute mean and variance for testing data.\n',getStat(my_data_test))









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
            #             nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),

            #                    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            #                    nn.BatchNorm2d(32), nn.ReLU(),
            #                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

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




transform_1 = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5181437, 0.50072074, 0.41316792), std=(0.21993916, 0.21960159, 0.2215463))
])

transform_2 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.51546204, 0.49846214, 0.41058794), std=(0.2197281, 0.21881661, 0.22134078))
])

transform_3 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.51805955, 0.5003706, 0.4121268), std=(0.21985851, 0.22019151, 0.2216588))
])


train_data = torchvision.datasets.ImageFolder(train_path,
                                              transform=transform_1,
                                              target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) )
val_data = torchvision.datasets.ImageFolder(val_path,transform=transform_2)
test_data = torchvision.datasets.ImageFolder(test_path,transform=transform_3)


resnet_18 = ResNet_18or34(2,2,2,2,10)

resnet_34 = ResNet_18or34(3,4,6,3,10)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



learn_rate = 0.001
num_epochs = 40
batch_size = 64

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

loss_fn = nn.CrossEntropyLoss()

lr_list_2 = []
train_result_list = []
val_result_list = []

best_f1_score = 0
# resnet=resnet_18
resnet = resnet_34

class_cnt = 10

warm_up_epochs = 5
warm_up_lr_start = 0.0001
warm_up_lr_end = learn_rate

optimizer = torch.optim.Adam(resnet.parameters(), lr=learn_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warm_up_epochs, eta_min=1e-5,
                                                       last_epoch=-1, verbose=False)

# warm up
for epoch in range(warm_up_epochs):
    start_time = time.time()

    dataloader = train_dataloader
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    resnet.train()

    optimizer.param_groups[0]['lr'] = warm_up_lr_start + (warm_up_lr_end - warm_up_lr_start) * (epoch) / (
                warm_up_epochs - 1)
    lr_list_2.append(optimizer.param_groups[0]['lr'])

    for batch, (X, y) in enumerate(dataloader):

        optimizer.zero_grad()
        pred = resnet(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 40 == 0:
            print(f"loss: {loss.item():>7f}  [{((batch + 1) * len(X)):>5d}/{size:>5d}]")

    train_result_list.append(train_loss / num_batches)

    dataloader = val_dataloader
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    confusion = np.zeros((class_cnt, class_cnt))
    resnet.eval()

    with torch.no_grad():
        for X, y in dataloader:
            pred = resnet(X)
            test_loss += loss_fn(pred, y).item()
            y_pred = pred.argmax(1)
            correct += (y_pred == y).type(torch.float).sum().item()

            for i in range(len(y)):
                confusion[[y_pred[i]], [y[i]]] += 1

    test_loss /= num_batches
    correct /= size

    precision_list = [confusion[i, i] / sum(confusion[i, :]) if sum(confusion[i, :]) != 0 else 0 for i in
                      range(class_cnt)]
    recall_list = [confusion[i, i] / sum(confusion[:, i]) if sum(confusion[:, i]) != 0 else 0 for i in range(class_cnt)]
    f1_score = np.mean([2 * (p * r) / (p + r) if p + r != 0 else 0 for p, r in zip(precision_list, recall_list)])
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}, F1-score: {(100 * f1_score):>0.1f}% \n")

    val_result_list.append([test_loss, correct, f1_score])

    end_time = time.time()
    print(f'Epoch {epoch} took {end_time - start_time} seconds!\n')

    if val_result_list[-1][2] > best_f1_score:
        best_f1_score = val_result_list[-1][2]
        best_resnet = copy.deepcopy(resnet)

# 余弦退火
for epoch in range(warm_up_epochs, num_epochs):
    start_time = time.time()

    dataloader = train_dataloader
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    resnet.train()

    scheduler.step()
    lr_list_2.append(optimizer.param_groups[0]['lr'])

    for batch, (X, y) in enumerate(dataloader):

        optimizer.zero_grad()
        pred = resnet(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 40 == 0:
            print(f"loss: {loss.item():>7f}  [{((batch + 1) * len(X)):>5d}/{size:>5d}]")

    train_result_list.append(train_loss / num_batches)

    dataloader = val_dataloader
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    confusion = np.zeros((class_cnt, class_cnt))
    resnet.eval()

    with torch.no_grad():
        for X, y in dataloader:
            pred = resnet(X)
            test_loss += loss_fn(pred, y).item()
            y_pred = pred.argmax(1)
            correct += (y_pred == y).type(torch.float).sum().item()

            for i in range(len(y)):
                confusion[[y_pred[i]], [y[i]]] += 1

    test_loss /= num_batches
    correct /= size

    precision_list = [confusion[i, i] / sum(confusion[i, :]) if sum(confusion[i, :]) != 0 else 0 for i in
                      range(class_cnt)]
    recall_list = [confusion[i, i] / sum(confusion[:, i]) if sum(confusion[:, i]) != 0 else 0 for i in range(class_cnt)]
    f1_score = np.mean([2 * (p * r) / (p + r) if p + r != 0 else 0 for p, r in zip(precision_list, recall_list)])
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}, F1-score: {(100 * f1_score):>0.1f}% \n")

    val_result_list.append([test_loss, correct, f1_score])

    end_time = time.time()
    print(f'Epoch {epoch} took {end_time - start_time} seconds!\n')

    if val_result_list[-1][2] > best_f1_score:
        best_f1_score = val_result_list[-1][2]
        best_resnet = copy.deepcopy(resnet)

torch.save(best_resnet.state_dict(), r'D:\work_files\task\best_resnet_weights.pth')






plt.figure(1)
plt.title('Learning Curve')
plt.plot( range(len(lr_list_2)) ,lr_list_2 )


epoch_idx_all=list(range(num_epochs))
train_result_list=[round(_,7) for _ in train_result_list]
val_loss_list=[round(_[0],7) for _ in val_result_list]
val_correct_list=[round(_[1],3) for _ in val_result_list]
val_F1_score_list=[round(_[2],3) for _ in val_result_list]

print('learn_rate:',learn_rate,'num_epochs:',num_epochs,'batch_size:',batch_size)

plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文

plt.figure(2)
plt.title('Loss Curve')
plt.plot(epoch_idx_all,train_result_list,marker='o',label='训练集')
plt.plot(epoch_idx_all,val_loss_list,marker='o',label='验证集')
plt.xlabel('epoch') # 横坐标标注
plt.ylabel('loss')
plt.legend() # 图例
plt.xticks(epoch_idx_all[::5]) # 横轴刻度
plt.grid(True) # 显示网格

plt.figure(3)
plt.title('Validation Data')
plt.plot(epoch_idx_all,val_correct_list,marker='o',label='准确率')
plt.plot(epoch_idx_all,val_F1_score_list,marker='o',label='F1-score')
plt.xlabel('epoch') # 横坐标标注
plt.legend() # 图例
plt.xticks(epoch_idx_all[::5]) # 横轴刻度
plt.grid(True) # 显示网格

plt.show()




record = pd.DataFrame({'epoch':epoch_idx_all,'lr':lr_list_2,'train_loss':train_result_list,'val_loss':val_loss_list,
                      'val_correct':val_correct_list,'val_F1_score':val_F1_score_list})
record.to_excel(r'D:\work_files\task\record_temp.xlsx')