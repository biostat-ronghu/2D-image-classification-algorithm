import random

import matplotlib.pyplot as plt

random.seed(-1)

# 数据路径
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Lambda

train_path = r"D:\work_files\task\train_data"
val_path = r"D:\work_files\task\val_data"
test_path = r"D:\work_files\task\test_data"

# 数据变换
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


#################### 数据加载
from torch.utils.data import DataLoader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size , shuffle= True, num_workers=0)
val_dataloader = DataLoader(val_data, batch_size=batch_size , shuffle= True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=batch_size , shuffle= True, num_workers=0)


#################### 模型加载
vit_base_patch16_224 = torchvision.models.vit_b_16( weights=torchvision.models.ViT_B_16_Weights.DEFAULT )
# 用imagenet_1K数据集(1000类)预训练,最后一层是torch.nn.linear(768,1000)
class_cnt=10
# 直接把(768,1000)改成(768,class_cnt)
vit_base_patch16_224.heads = torch.nn.Linear(768,class_cnt,bias=True)
# 后面接一层(1000,class_cnt)
# vit_base_patch16_224.heads.add_module('add_my_linear',torch.nn.Linear(1000,class_cnt,bias=True))
# 预训练模型参数冻结
for name_temp,parameter_temp in vit_base_patch16_224.named_parameters():
    if name_temp.startswith('heads'):
        parameter_temp.requires_grad = True
    else:
        parameter_temp.requires_grad = False

    # if name_temp.startswith('heads.add_my_linear'):
    #     parameter_temp.requires_grad = True
    # else:
    #     parameter_temp.requires_grad = False

# print(vit_base_patch16_224)
# for name_temp,_ in vit_base_patch16_224.named_parameters():
#         print(f'参数{name_temp}是否允许更新:{_.requires_grad}')



#################### 模型微调
import time
import copy
import numpy as np

learn_rate = 0.001
num_epochs = 10

loss_fn = torch.nn.CrossEntropyLoss()

lr_list_2=[]
train_result_list=[]
val_result_list=[]

best_f1_score=0
vit_model = vit_base_patch16_224

warm_up_epochs = 5
warm_up_lr_start = 0.0001
warm_up_lr_end = learn_rate

optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, vit_model.parameters()),lr=learn_rate )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-warm_up_epochs, eta_min=1e-5, last_epoch= -1, verbose=False)



# warm up
for epoch in range(warm_up_epochs):
    start_time = time.time()

    dataloader = train_dataloader
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    vit_model.train()

    optimizer.param_groups[0]['lr'] = warm_up_lr_start + (warm_up_lr_end - warm_up_lr_start) * (epoch) / (
                warm_up_epochs - 1)
    lr_list_2.append(optimizer.param_groups[0]['lr'])

    for batch, (X, y) in enumerate(dataloader):

        optimizer.zero_grad()
        pred = vit_model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 20 == 0:
            print(f"loss: {loss.item():>7f}  [{((batch + 1) * len(X)):>5d}/{size:>5d}]")
            print(f"each batch took {(time.time() - start_time)/(batch+1)} seconds!")

    train_result_list.append(train_loss / num_batches)

    dataloader = val_dataloader
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    confusion = np.zeros((class_cnt, class_cnt))
    vit_model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            pred = vit_model(X)
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
        best_vit = copy.deepcopy(vit_model)


# 余弦退火
for epoch in range(warm_up_epochs, num_epochs):
    start_time = time.time()

    dataloader = train_dataloader
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    vit_model.train()

    scheduler.step()
    lr_list_2.append(optimizer.param_groups[0]['lr'])

    for batch, (X, y) in enumerate(dataloader):

        optimizer.zero_grad()
        pred = vit_model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 20 == 0:
            print(f"loss: {loss.item():>7f}  [{((batch + 1) * len(X)):>5d}/{size:>5d}]")
            print(f"each batch took {(time.time() - start_time)/(batch+1)} seconds!")

    train_result_list.append(train_loss / num_batches)

    dataloader = val_dataloader
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    confusion = np.zeros((class_cnt, class_cnt))
    vit_model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            pred = vit_model(X)
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
        best_resnet = copy.deepcopy(vit_model)


torch.save(best_vit, r'D:\work_files\task\best_vit.pth')



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



import pandas as pd
record = pd.DataFrame({'epoch':epoch_idx_all,'lr':lr_list_2,'train_loss':train_result_list,'val_loss':val_loss_list,
                      'val_correct':val_correct_list,'val_F1_score':val_F1_score_list})
record.to_excel(r'D:\work_files\task\record_temp_vit.xlsx')







# vit_model.eval()
#
# dataloader = test_dataloader
# size = len(dataloader.dataset)
# correct = 0
# confusion = np.zeros((class_cnt, class_cnt))
#
# start_time = time.time()
# with torch.no_grad():
#     for batch, (X, y) in enumerate(dataloader):
#         pred = vit_model(X)
#         y_pred = pred.argmax(1)
#         correct += (y_pred == y).type(torch.float).sum().item()
#
#         for i in range(len(y)):
#             confusion[[y_pred[i]], [y[i]]] += 1
#
#         print(f"each batch took {(time.time() - start_time) / (batch + 1)} seconds!")
#
# correct /= size
#
# precision_list = [confusion[i, i] / sum(confusion[i, :]) if sum(confusion[i, :]) != 0 else 0 for i in range(class_cnt)]
# recall_list = [confusion[i, i] / sum(confusion[:, i]) if sum(confusion[:, i]) != 0 else 0 for i in range(class_cnt)]
# f1_score = np.mean([2 * (p * r) / (p + r) if p + r != 0 else 0 for p, r in zip(precision_list, recall_list)])
# print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, F1-score: {(100 * f1_score):>0.1f}% \n")

