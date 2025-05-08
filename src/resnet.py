from torch import nn
import os

import torch
from PIL import Image
from torch.utils import data
import numpy as np

from torchvision import transforms as T
import torch

import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torchvision import transforms, datasets
import os, PIL, pathlib, warnings
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
from PIL import Image

from sklearn.metrics import f1_score
from sklearn import metrics
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])

import torchvision.transforms as transforms
from torchvision import transforms, datasets

train_transforms = transforms.Compose([
    transforms.Resize([224, 224]),  # 将图片统一尺寸
    transforms.RandomHorizontalFlip(),  # 将图片随机水平翻转
    transforms.ToTensor(),  # 将图片转换为tensor
    transforms.Normalize(  # 标准化处理—>转换为正态分布，使模型更容易收敛
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
test_transforms = transforms.Compose([
    transforms.Resize([224, 224]),  # 将图片统一尺寸
    #transforms.RandomHorizontalFlip(),  # 将图片随机水平翻转
    transforms.ToTensor(),  # 将图片转换为tensor
    transforms.Normalize(  # 标准化处理—>转换为正态分布，使模型更容易收敛
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class ResNetblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetblock, self).__init__()
        self.blockconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels * 4)
        )
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        residual = x
        out = self.blockconv(x)
        if hasattr(self, 'shortcut'):  # 如果self中含有shortcut属性
            residual = self.shortcut(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, block, num_classes=2):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2)
        )
        self.in_channels = 64
        # ResNet50中的四大层，每大层都是由ConvBlock与IdentityBlock堆叠而成
        self.layer1 = self.make_layer(ResNetblock, 64, 3, stride=1)
        self.layer2 = self.make_layer(ResNetblock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResNetblock, 256, 6, stride=2)
        self.layer4 = self.make_layer(ResNetblock, 512, 3, stride=2)

        self.avgpool = nn.AvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 4, num_classes)

    # 每个大层的定义函数
    def make_layer(self, block, channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * 4

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 训练集的大小
    num_batches = len(dataloader)  # 批次数目

    train_loss, train_acc = 0, 0  # 初始化训练损失和正确率
    predicted_label = torch.empty(0).to(device)
    true_label = torch.empty(0).to(device)

    for x, y in dataloader:  # 获取图片及其标签
        x, y = x.to(device), y.to(device)

        # 计算预测误差
        pred = model(x)  # 网络输出
        loss = loss_fn(pred, y)  # 计算网络输出和真实值之间的差距，二者差值即为损失

        # 反向传播
        optimizer.zero_grad()  # grad属性归零
        loss.backward()  # 反向传播
        optimizer.step()  # 每一步自动更新

        # 记录acc与loss
        #print(pred.argmax(1))
        #print(y)
        #print((pred.argmax(1) == y).type(torch.float).sum().item())
        #print(predicted_label)
        #print(pred.argmax(1))
        predicted_label = torch.cat((predicted_label,pred.argmax(1)),dim=0)
        #print(predicted_label)
        true_label = torch.cat((true_label,y.argmax(1)),dim=0)

        train_acc += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        train_loss += loss.item()
    f1 = f1_score(predicted_label.cpu().numpy(),true_label.cpu().numpy())
    train_acc /= size

    train_loss /= num_batches

    return train_acc, train_loss,f1


# 测试函数
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 测试集的大小
    num_batches = len(dataloader)  # 批次数目
    test_loss, test_acc = 0, 0

    # 当不进行训练时，停止梯度更新，节省计算内存消耗
    predicted_label = torch.empty(0).to(device)
    true_label = torch.empty(0).to(device)
    predicted_val = torch.empty(0).to(device)
    target0 = 0
    target1 = 0
    TN = 0
    TP = 0
    step = 0
    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)
            step += 1
            #
            # 计算loss
            target_pred = model(imgs)
            #print(target_pred,target)
            loss = loss_fn(target_pred, target)
            #print(target_pred.argmax(1),target.argmax(1))
            predicted_label = torch.cat((predicted_label, target_pred.argmax(1)), dim=0)
            predicted_val = torch.cat((predicted_val,target_pred.max(1).values),dim=0)

            true_label = torch.cat((true_label, target.argmax(1)), dim=0)


            test_loss += loss.item()
            test_acc += (target_pred.argmax(1) == target.argmax(1)).type(torch.float).sum().item()
            #print(type(target_pred.argmax(1)))
            TN += ((target_pred.argmax(1) == target.argmax(1)) & (target.argmax(1) == 0)).type(torch.float).sum().item()
            #print(type(target_pred.argmax(1)))
            TP += ((target_pred.argmax(1) == target.argmax(1)) & (target.argmax(1) == 1)).type(torch.float).sum().item()
            target0 += (target.argmax(1) == 0).type(torch.float).sum().item()
            target1 += (target.argmax(1) == 1).type(torch.float).sum().item()

    f1 = f1_score(predicted_label.cpu().numpy(),true_label.cpu().numpy())
    fpr, tpr, thresholds = metrics.roc_curve(true_label.cpu().numpy(), predicted_val.cpu().numpy())
    test_acc /= size
    test_loss /= num_batches
    P0 = TN / target0
    P1 = TP / target1


    return test_acc, test_loss,f1,fpr, tpr, thresholds, P0, P1

def load(model,weight):
    pretrained_weight_path = weight
    # 加载预训练权重，返回类型是字典
    pretrained_dict = torch.load(pretrained_weight_path)
    net = model
    # 加载自定义网络模型权重
    model_dict = net.state_dict()
    # 判断预训练权重模型和自定义网络的模型参数，如果key和对应shape都相同则取出，否则就去掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and (v.shape == model_dict[k].shape)}
    # 更新修改后的参数
    model_dict.update(pretrained_dict)
    # 并重新让模型加载参数dict
    net.load_state_dict(model_dict, strict=True)
    return net

def StrToLabel(Str):
    # print(Str)
    label = []

    if Str == '0' or Str == '1':
        label.append(1)
        label.append(0)

    elif Str == '2' or Str == '3':
        label.append(0)
        label.append(1)

    return label


def LabelToStr(Label):
    Str = ""
    for i in Label:
        if i <= 9:
            Str += chr(ord('0') + i)
        elif i <= 35:
            Str += chr(ord('a') + i - 10)
        else:
            Str += chr(ord('A') + i - 36)
    return Str
class BCI(data.Dataset):
    def __init__(self, root, train=True):
        self.imgPath = [os.path.join(root, img) for img in os.listdir(root)]
        """

        self.imgPath = []
        for img in os.listdir(root):

            if 'fake' in img:
                self.imgPath.append(os.path.join(root, img))
        """
        if train:
            self.transform = T.Compose([
                transforms.Resize([224, 224]),  # 将图片统一尺寸
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),# 将图片随机水平翻转，推理时无需增强，保存时用acc做依据，当数据不平衡时用f1 score或roc，补充一个垂直翻转增强，vertical，
                # 控制图像被数据增强的概率,选择p=0.3，保留最好的model，用resnet评估RDDM生成结果
                transforms.ToTensor(),  # 将图片转换为tensor
                transforms.Normalize(  # 标准化处理—>转换为正态分布，使模型更容易收敛，不需要重新计算
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = T.Compose([
                transforms.Resize([224, 224]),  # 将图片统一尺寸
                #transforms.RandomHorizontalFlip(),
                # 将图片随机水平翻转，推理时无需增强，保存时用acc做依据，当数据不平衡时用f1 score或roc，补充一个垂直翻转增强，vertical，
                # 控制图像被数据增强的概率,选择p=0.3，保留最好的model，用resnet评估RDDM生成结果
                transforms.ToTensor(),  # 将图片转换为tensor
                transforms.Normalize(  # 标准化处理—>转换为正态分布，使模型更容易收敛，不需要重新计算
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])




    def __getitem__(self, index):
        img_path = self.imgPath[index]
        #print(img_path)
        label = img_path.split('/')[-1].split('.')[0].split('_')[-1][0] #获取图片标签
        #print(img_path)
        #label = img_path.split('/')[-1].split('.')[0].split('_')[2][0]
        #print(label)

        label_tensor = torch.Tensor(StrToLabel(label))
        data=Image.open(img_path)
        data = self.transform(data)  # 使用PLT打开图片文件
        return data, label_tensor

    def __len__(self):
        return len(self.imgPath)


training = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':

    if training:

        model = ResNet50(block=ResNetblock,num_classes=2).to(device)
        train_path='/mnt/data/BCI/train/IHC'
        val_path='/mnt/data/BCI/test/IHC'
        train_dataset = BCI(train_path)
        val_dataset = BCI(val_path)
        train_dl = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, pin_memory=True,
                                       num_workers=4)
        test_dl = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True, pin_memory=True,
                                     num_workers=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 创建优化器，并设置学习率
        loss_fn = nn.CrossEntropyLoss()  # 创建损失函数

        epochs = 100

        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        best_f1 = 0  # 设置一个最佳准确率，作为最佳模型的判别指标

        for epoch in range(epochs):

            model.train()
            epoch_train_acc, epoch_train_loss,epoch_f1 = train(train_dl, model, loss_fn, optimizer)

            model.eval()
            epoch_test_acc, epoch_test_loss,epoch_test_f1,fpr, tpr, thresholds, P0, P1 = test(test_dl, model, loss_fn)

            # 保存最佳模型到J1_model
            if epoch_test_f1 > best_f1:
                best_f1 = epoch_test_f1
                J1_model = copy.deepcopy(model)

            train_acc.append(epoch_train_acc)
            train_loss.append(epoch_train_loss)
            test_acc.append(epoch_test_acc)
            test_loss.append(epoch_test_loss)

            # 获取当前学习率
            lr = optimizer.state_dict()['param_groups'][0]['lr']

            template = ('Epoch:{:2d},Train_acc:{:.1f}%,Train_loss:{:.3f},Test_acc:{:.1f}%,Test_loss:{:.3f},Lr:{:.2E}')
            print('f1:{},test_f1:{}'.format(epoch_f1,epoch_test_f1))
            print(template.format(epoch + 1, epoch_train_acc * 100, epoch_train_loss,
                                  epoch_test_acc * 100, epoch_test_loss, lr))

            # 保存最佳模型到文件中
            PATH = r'/mnt/data/result_ge47nej/resnet/ckpt_full_model/model2.pth'
            torch.save(J1_model.state_dict(), PATH)
    else:
        model = ResNet50(block=ResNetblock,num_classes=2).to(device)
        weight_path = '/mnt/data/result_ge47nej/resnet/ckpt_full_model/model2.pth'
        model = load(model,weight_path)

        #train_path = '/mnt/data/BCI/train/IHC'
        val_path = '/mnt/data/result_ge47nej/results_translation_test/512_imagesize_SFS_sample_checkpoint_44_1024_ablation_step_50'
        #train_dataset = BCI(train_path)
        val_dataset = BCI(val_path,train=False)
        #train_dl = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, pin_memory=True,
        #                      num_workers=4)
        test_dl = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, pin_memory=True,
                             num_workers=4)
        #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 创建优化器，并设置学习率
        loss_fn = nn.CrossEntropyLoss()  # 创建损失函数

        for epoch in range(1):
            model.eval()
            epoch_test_acc, epoch_test_loss,epoch_test_f1,fpr, tpr, thresholds, P0, P1 = test(test_dl, model, loss_fn)
            p0_c = 0.5840
            p1_c = 0.9659
            delta_p0 = p0_c - P0
            delta_p1 = p1_c - P1
            avg_deg = (delta_p1 + delta_p0) / 2
            SFS = (epoch_test_acc + (1 - avg_deg)) / 2

            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Roc curve')
            plt.legend(loc="lower right")
            plt.savefig('/mnt/data/result_ge47nej/resnet/roc_plot.png')
            plt.show()
            print(val_path)
            print('test_f1:{}，P0:{}, P1:{},acc,:{},SFS:{}'.format(epoch_test_f1,P0,P1,epoch_test_acc,SFS))


