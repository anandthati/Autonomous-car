import os
import torch
from torch.autograd import Variable
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import torch.nn.functional as F

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=3)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3)
        self.conv3 = nn.Conv2d(32,16,kernel_size=3)
        self.conv4 = nn.Conv2d(16,8,kernel_size=3)
        self.conv5 = nn.Conv2d(8,4,kernel_size=3)
        self.fc = nn.Linear(100,5)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,img):
        img_size = img.size(0)
        c1 = F.relu(self.maxpool(self.conv1(img)))
        c2 = F.relu(self.maxpool(self.conv2(c1)))
        c3 = F.relu(self.maxpool(self.conv3(c2)))
        c4 = F.relu(self.maxpool(self.conv4(c3)))
        c5 = F.relu(self.maxpool(self.conv5(c4)))
        c5 = c5.view(img_size, -1)
        c6 = self.fc(c5)
        return c6