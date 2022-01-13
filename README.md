# 手写数字识别

> 代码：[手写数字识别代码](https://paste.org.cn/aFdjZdFYul)
>
> 来源：[github](https://github.com/liucj97/CNN_MNIST_recognition_by_Pytorch)

使用 `matplotlib`来可视化，需要修改中文字体：

```python
# 用来正常显示中文标签
matplotlib.rcParams["font.family"] = 'Arial Unicode MS'
```

### 训练集

NNIST训练集来源：`torchvision.datasets.MNIST`

```python
# 载入MNIST训练集
train_dataset = datasets.MNIST(
    root='.',  # 数据集下载目录
    train=True,  # 训练集标记
    transform=transforms.ToTensor(),  # 转为Tensor变量
    download=download)
```

Pytorch中提供了一种叫做 `DataLoader`的方法来让我们进行训练，该方法自动将数据集打包成为迭代器，能够让我们很方便地进行后续的训练处理。

```python
train_loader = DataLoader(
    dataset=train_dataset,  # 数据集加载
    shuffle=True,  # 随机打乱数据
    batch_size=batch_size)  # 批处理量100
```

`train_loader	` 包含img和label都是torch.Tensor类型。

#### label的格式：

- 前4个字节（第0～3个字节）是魔数2049（int型，0x00000801, 大端）；
- 再往后4个字节（第4～7个字节）是标签的个数：60000或10000；
- 再往后每1个字节是一个无符号型的数，值为0～9。

#### img的格式：

- 前4个字节（第0～3个字节）是魔数2051（int型，0x00000803, 大端）;
- 再往后4个字节（第4～7个字节）是图像的个数：60000或10000（第1个维度）；
- 再往后4个字节（第8～11个字节）是图像在高度上由多少个像素组成（第2个维度，高28个像素）；
- 再往后4个字节（第12～15个字节）是图像在宽度上由多少个像素组成（第3个维度，宽28个像素）；
- 再往后是一个三维数组，表示10000个或60000个分辨率28x28的灰度图像，一句话来说就是10000x28x28个像素，每个像素的值为0～255（0是背景，为白色；255是黑色）。
- 可以使用matplotlib.pyplot来显示其中一个数字的图像：

```python
import matplotlib.pyplot as plt
plt.imshow(images[0])
plt.show()
```

### 展示图像

`make_grid`的作用是将若干幅图像拼成一幅图像

<img src="https://s2.loli.net/2022/01/01/sjPkpwh5fAGX2S4.png" alt="image-20220101212609435" style="zoom:33%;" />

```python
def imshow(img, title=None):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# 存入迭代器 展示部分数据
dataiter = iter(train_loader)
batch = next(dataiter)
if show_pic: # 超参数，是否需要可视化
    imshow(make_grid(batch[0], nrow=10, padding=2, pad_value=1), '训练集部分数据')
```

### 初始化卷积神经网络

`MNIST_Network`是 `MNIST_Network`的继承类，**class里的内容：`__init_()` 和 `forward()` 还看不懂**。

初始化神经网络

```python
net = MNIST_Network() # 初始化，调用__init__()函数
if use_gpu:  # CUDA GPU加速
    net = net.cuda()
```

### 训练部分

对于简单的多分类任务，我们可以使用 `交叉熵损失`来作为损失函数，而对于迭代器而言，我们可以使用 `Adam迭代器`。

训练 `epochs_num`次，`enumerate(seq, [start=0])	`函数用于将可遍历对象（列表，元组，字符串）组合为一个索引列表，同时列出下标和数据。

**为什么向前传播是 `output = net(img)`，而不是调用forward()函数？**

可以这样记录正确次数：`correct_cnt += (predict == label).sum()`。

```python
if is_train:
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
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
        for i, data in enumerate(train_loader, 0): # (元素，下标)
            img, label = data # 图像和标签两个tensor
            if use_gpu:  #CUDA GPU加速
                img, label = img.cuda(), label.cuda()

            optimizer.zero_grad()  # 清除网络状态（模型的梯度）
            output = net(img)  # 前向传播
            loss = criterion(output, label)  # 计算损失函数
            loss.backward()  # 反向传播求梯度
            optimizer.step()  # 使用迭代器更新模型权重

            _, predict = torch.max(output, 1)
            correct_cnt += (predict == label).sum()  # 预测值与实际值比较，记录正确次数

            # 存储损失值与精度
            if i % record_interval == record_interval - 1:
                counter_temp += record_interval * batch_size
                counter.append(counter_temp)
                loss_history.append(loss.item())
                correct_history.append(correct_cnt.float().item() /
                                       (record_interval * batch_size))
                correct_cnt = 0
        print("迭代次数 {}\n 当前损失函数值 {}\n".format(epoch, loss.item()))
```

### 绘制损失函数与精度曲线

```python
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
```

### 模型参数

存储模型，下次可以直接调用

```python
state = {'net': net.state_dict()}
torch.save(net.state_dict(), './modelpara.pth')
```

加载模型参数

```python
if use_gpu:
    net.load_state_dict(torch.load('./modelpara.pth'))
else:
    net.load_state_dict(torch.load('./modelpara.pth', map_location='cpu'))
```

### 测试集

NNIST测试集来源：`torchvision.datasets.MNIST`

```python
test_dataset = datasets.MNIST(root='.',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=download)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          shuffle=True,
                                          batch_size=batch_size)
```

展示部分数据：

```python
dataiter = iter(test_loader)
batch = next(dataiter)
if show_pic:
    imshow(
        torchvision.utils.make_grid(batch[0], nrow=10, padding=2, pad_value=1),
        '测试集部分数据')
```

> `next()`使用方法：
>
> ```python
> # 首先获得Iterator对象:
> it = iter([1, 2, 3, 4, 5])
> while True:
>     try: # 获得下一个值:
>         x = next(it)
>         print(x)
>     except StopIteration: # 遇到StopIteration就退出循环
>         break
> '''
> 1
> 2
> 3
> 4
> 5
> '''
> ```

训练集预测测试：

```python
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
```

显示部分测试结果与输出准确率和时间：

```python
if show_pic: 
    imshow(
        torchvision.utils.make_grid(img[75:100].cpu(),
                                    nrow=5,
                                    padding=2,
                                    pad_value=1),
        '25张测试结果:\n' + str(predict[75:100].cpu().numpy()))w

# 输出测试准确率及时间
print('MNIST测试集识别准确率= {:.2f}'.format(correct.cpu().numpy() / len(test_dataset) * 100) + '%')
print('10000张识别时间= {:.3f}'.format(end - start) + ' s')
```

### 实现流程

<img src="https://s2.loli.net/2022/01/02/bFcfD6547qtGvs3.png" alt="image-20220102183001450" style="zoom:50%;" />

### 运行效果

<img src="https://s2.loli.net/2022/01/02/5VTANvQI9XOEJGu.png" alt="image-20220102180911176" style="zoom:50%;" />

![image-20220102200455822](https://s2.loli.net/2022/01/02/IaXwUlso5MjeQCf.png)

<img src="https://s2.loli.net/2022/01/02/4BMJVId2Nl8WoLY.png" alt="image-20220102180932998" style="zoom:50%;" />

<img src="https://s2.loli.net/2022/01/02/qtS5j7E8TsKL6zH.png" alt="image-20220102180945947" style="zoom:50%;" />

<img src="https://s2.loli.net/2022/01/02/sf2ekzUo1rPxTl5.png" alt="image-20220102181034556" style="zoom:50%;" />
