import torch
import cv2
import sys
sys.path.append('../hackathon')
from cnn_model import MyCNN
PATH='./CNN.pt'
Outputs = {'0':'Left Turn','1':'Traffic Light','2':'U turn','3':'Right Turns','4':'Stop'}

img = cv2.imread('/home/eindhan/catkin_workspace/src/hackathon/owndata/camera_img_right_3.jpeg',0)
img = cv2.resize(img, (224,224))
img = torch.from_numpy(img).type(torch.FloatTensor)
imgs = torch.unsqueeze(img,0)
imgs = torch.unsqueeze(imgs,0)
the_model = MyCNN()
the_model.load_state_dict(torch.load(PATH))
the_model.eval()
#the_model = torch.load(PATH)
out = the_model(imgs).data.max(1, keepdim=True)[1][0][0].data.cpu().numpy()
print 'Action: ',Outputs[str(out)]

