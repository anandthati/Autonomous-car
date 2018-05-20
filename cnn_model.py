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

#DIR = {'left turns':0,'traffic_light':1,'u_turn':2,'right_turns':3,'stop':4}
DIR = {'left':0,'right':1,'uturn':2,'stop':3}
Outputs = {'0':'Left Turn','1':'Right Light','2':'U turn','3':'Stop'}
SUB_DIR = ['images', 'augmentedImages']
image_name = []
image_tag = []
for key,value in DIR.items():
    #for sd in SUB_DIR:
    for img in os.listdir('data/'+key):
        if img != '.DS_Store':
            image_name.append(key+'/'+img)
            image_tag.append(value)
df = pd.DataFrame({'Name':image_name,'Target':image_tag})
df.to_csv('ImageDataset.csv',index=None)
Image_Train, Image_Test, Label_Train, Label_Test = train_test_split(image_name,image_tag,test_size = 0.3)

class MyCustomDataset(Dataset):
    def __init__(self,imgs,targets):
        # stuff
        self.imgpath = imgs
        self.label = targets
    def __getitem__(self, index):
        img = cv2.imread('data/'+self.imgpath[index],0)
        img = cv2.resize(img, (224,224))
        img = torch.from_numpy(img).type(torch.FloatTensor)
        img = torch.unsqueeze(img,0)
        # stuff
        label = self.label[index]
        #print(label)
        return (img, label)

    def __len__(self):
        return len(self.imgpath) # of how many examples(images?) you have

dataset = MyCustomDataset(Image_Train,Label_Train)
train_loader = DataLoader(dataset,
                          batch_size=10,
                          shuffle=True,
                         )
test_dataset = MyCustomDataset(Image_Test,Label_Test)
test_loader = DataLoader(test_dataset,
                          batch_size=10,
                          shuffle=True,
                         )

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

model = MyCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

def Train(epoch):
    model.train()
    for index,(img,target) in enumerate(train_loader):
        img,target = Variable(img),Variable(target)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        if index % 10 == 0:
            print(output.data.max(1, keepdim=True)[1],target)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, index * len(img), len(train_loader.dataset),
                100. * index / len(train_loader), loss.data[0]))

def Test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for e in range(1,1):
    Train(e)
    Test()

img = cv2.imread('/home/eindhan/Desktop/AIML_Team_49/signs_data/right_turns/augmentedImages/add1_r27.png',0)
img = cv2.resize(img, (224,224))
img = torch.from_numpy(img).type(torch.FloatTensor)
imgs = torch.unsqueeze(img,0)
imgs = torch.unsqueeze(imgs,0)
out = model(imgs).data.max(1, keepdim=True)[1][0][0].data.cpu().numpy()
#print 'Action: ',Outputs[str(out)]
    
#torch.save(model, './CNN')
PATH='./CNN'
torch.save(model.state_dict(), PATH)