import torch 
from torch import nn
import torch.nn.functional as F

class NN(nn.Module):
    name = "znn.pth"

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size = 3, stride = 2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size = 3, stride = 2)
        self.conv6 = nn.Conv2d(256, 256, kernel_size = 3, stride = 2)
        self.conv7 = nn.Conv2d(256, 256, kernel_size = 3, stride = 2)
        self.conv8 = nn.Conv2d(256, 256, kernel_size = 1)
        self.conv9 = nn.Conv2d(256, 128, kernel_size = 1)
        self.conv10 = nn.Conv2d(128, 64, kernel_size = 1)

    def forward(self, x): 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.conv10(x)

        return x
    def loadModel():
        model = NN()
        try:
            model.load_state_dict(torch.load(NN.name))
        except:
            pass
            
        return model