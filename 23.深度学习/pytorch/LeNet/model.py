import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        ##如果输入为28*28*1,则卷积核为 5*5*1*6
        self.conv1 = nn.Conv2d(1, 6, 5, 1,2)
        self.sigmoid = nn.Sigmoid()  ##sigmoid每次卷积都需要
        self.avgPool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.avgPool2 = nn.AvgPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.avgPool1(x)

        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.avgPool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    ##输入图像为28*28*1。  参数个数=输出通道数*（输入通道*卷积核高*卷积核宽+1）=6*（1*5*5+1）。这里的+1是因为偏置个数
    summary(model, (1, 28, 28))