'''
基于Pytorch的卷积神经网络MNIST手写数字识别
https://github.com/liucj97/CNN_MNIST_recognition_by_Pytorch
时间:2020年4月
配置方式:见README.txt
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
import time

# 设置plot中文字体
matplotlib.rcParams["font.family"] = 'Arial Unicode MS' #用来正常显示中文标签


# 辅助函数-展示图像
def imshow(img, title=None):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


# 设定超参数及常数
learning_rate = 0.0005  #学习率
batch_size = 100  #批处理量
epochs_num = 10  #训练迭代次数
download = False  #数据集加载方式
use_gpu = 0  #CUDA GPU加速  1:使用  0:禁用
is_train = 0  #训练模型  1:重新训练     0:加载现有模型
show_pic = 1  #图像展示  1:展示过程图像  0:关闭图像显示

# 载入MNIST训练集
train_dataset = datasets.MNIST(
    root='.',  # 数据集下载目录
    train=True,  # 训练集标记
    transform=transforms.ToTensor(),  # 转为Tensor变量
    download=download)

train_loader = DataLoader(
    dataset=train_dataset,  # 数据集加载
    shuffle=True,  # 随机打乱数据
    batch_size=batch_size)  # 批处理量100

# 存入迭代器 展示部分数据
dataiter = iter(train_loader)
batch = next(dataiter)
if show_pic:
    imshow(make_grid(batch[0], nrow=10, padding=2, pad_value=1), '训练集部分数据')


# 初始化卷积神经网络
class MNIST_Network(nn.Module):
    def __init__(self):
        super(MNIST_Network, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # 卷积层
        self.relu1 = nn.ReLU()  # 激活函数ReLU
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 最大池化层

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # 卷积层
        self.relu2 = nn.ReLU()  # 激活函数ReLU
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 最大池化层

        self.fc3 = nn.Linear(7 * 7 * 64, 1024)  # 全连接层
        self.relu3 = nn.ReLU()  # 激活函数ReLU

        self.fc4 = nn.Linear(1024, 10)  # 全连接层
        self.softmax4 = nn.Softmax(dim=1)  # Softmax层

    # 前向传播
    def forward(self, input1):
        x = self.conv1(input1)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size()[0], -1)
        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.softmax4(x)
        return x


# 初始化神经网络
net = MNIST_Network()
if use_gpu:  # CUDA GPU加速
    net = net.cuda()

if is_train:
    lossF = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate)  # 使用adam算法进行训练

    counter = [] # 训练数据的数量
    loss_history = [] # 存储识别损失
    correct_history = [] # 存储识别精度
    correct_cnt = 0 # 预测正确次数
    counter_temp = 0 # 总预测次数
    record_interval = 100 # 记录间隔
    # 多次迭代训练网络
    for epoch in range(0, epochs_num):
        # 开始对训练集train_loader进行迭代
        processBar = tqdm(train_loader, unit='step') # 构建tqdm进度条
        for step, (img, label) in enumerate(processBar):
            if use_gpu:  #CUDA GPU加速
                img, label = img.cuda(), label.cuda()

            optimizer.zero_grad()  # 清除网络状态（模型的梯度）
            output = net(img)  # 前向传播，测试部分仅需要先前传播即可
            loss = lossF(output, label)  # 计算损失函数
            loss.backward()  # 反向传播求梯度
            optimizer.step()  # 使用迭代器更新模型权重

            _, predict = torch.max(output, 1)
            correct_cnt += (predict == label).sum()  # 预测值与实际值比较，记录正确次数
            accuracy = (predict == label).sum() / label.shape[0] # 正确率
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % (epoch, epochs_num, loss.item(), accuracy.item()))

            # 存储损失值与精度
            if step % record_interval == record_interval - 1:
                counter_temp += record_interval * batch_size
                counter.append(counter_temp)
                loss_history.append(loss.item())
                correct_history.append(correct_cnt.float().item() / (record_interval * batch_size))
                correct_cnt = 0
        print("迭代次数 {} 当前损失函数值 {}\n".format(epoch, loss.item()))
    processBar.close()
    # 绘制损失函数与精度曲线
    if show_pic:
        plt.figure(figsize=(20, 10), dpi=80)
        plt.subplot(211)
        plt.plot(counter, loss_history)
        plt.xlabel('训练张数')
        plt.ylabel('损失函数值')
        plt.title('损失函数曲线')
        plt.subplot(212)
        plt.plot(counter, correct_history)
        plt.xlabel('训练张数')
        plt.ylabel('精确度')
        plt.title('精确度曲线')
        plt.show()

    # 存储模型参数
    state = {'net': net.state_dict()}
    torch.save(net.state_dict(), './modelpara.pth')

# 加载模型参数
if use_gpu:
    net.load_state_dict(torch.load('./modelpara.pth'))
else:
    net.load_state_dict(torch.load('./modelpara.pth', map_location='cpu'))

# MNIST测试集准确率测试
test_dataset = datasets.MNIST(root='.',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=download)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          shuffle=True,
                                          batch_size=batch_size)
# 存入迭代器 展示部分数据
dataiter = iter(test_loader) # 迭代器
batch = next(dataiter) # 迭代器的下一个项目，即img
if show_pic:
    imshow(
        torchvision.utils.make_grid(batch[0], nrow=10, padding=2, pad_value=1),
        '测试集部分数据')

# 训练集预测测试
start = time.time()
correct = 0
for i, data in enumerate(test_loader, 0):
    img, label = data
    if use_gpu:  #CUDA GPU加速
        img, label = img.cuda(), label.cuda()
    output = net(img)  # 前向传播
    _, predict = torch.max(output, 1)
    correct += (predict == label).sum()  # 预测值与实际值比较
end = time.time()
# 显示部分测试结果
if show_pic: 
    imshow(
        torchvision.utils.make_grid(img[75:100].cpu(),
                                    nrow=5,
                                    padding=2,
                                    pad_value=1),
        '25张测试结果:\n' + str(predict[75:100].cpu().numpy()))

# 输出测试准确率及时间
print('MNIST测试集识别准确率= {:.2f}'.format(correct.cpu().numpy() / len(test_dataset) * 100) + '%')
print('10000张识别时间= {:.3f}'.format(end - start) + ' s')
