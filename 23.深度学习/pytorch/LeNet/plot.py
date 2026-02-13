from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt


##下载数据并转成张量
train_data = FashionMNIST(root='./data',
                          train=True,  #只下载训练集不下载数据集
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]), ##数据转换成张量形式
                          download=True)

##使用pytorch Data加载数据
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

## enumerate(train_loader) 会返回(批次编号,批次数据），这里的step就是批次，(b_x, b_y)包含数据和标签
for step, (b_x, b_y) in enumerate(train_loader):
    # 获得一个Batch的数据
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
batch_y = b_y.numpy()  # 将张量转换成Numpy数组
class_label = train_data.classes  # 训练集的标签
# print(class_label)
print("The size of batch in train data:", batch_x.shape)  # 每个mini-batch的维度是64*224*224

# 可视化一个Batch的图像
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()