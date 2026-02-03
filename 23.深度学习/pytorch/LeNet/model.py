import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        ##如果输入为28*28*1,则卷积核为 5*5*1*6
        self.conv1 = nn.Conv2d(1, 6, 5, 2)
        self.sigmoid = nn.Sigmoid()  ##sigmoid要几次？？？
        self.avgPool1 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.avgPool2 = nn.AvgPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

