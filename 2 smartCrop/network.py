import torch 
from torch import nn
import torch.nn.functional as F

class NN(nn.Module):
    name = "znn.pth"

    def __init__(self):
        super().__init__()

        self.conv1a = nn.Conv2d(3, 80, kernel_size = 3)
        self.conv1b = nn.Conv2d(80, 80, kernel_size = 3, stride = 2)

        self.conv2a = nn.Conv2d(80, 120, kernel_size = 3)
        self.conv2b = nn.Conv2d(120, 120, kernel_size = 3, stride = 2)

        self.conv3a = nn.Conv2d(120, 160, kernel_size = 3)
        self.conv3b = nn.Conv2d(160, 160, kernel_size = 3, stride = 2)
        self.bn3 = nn.BatchNorm2d(160)

        self.conv4a = nn.Conv2d(160, 200, kernel_size = 3)
        self.conv4b = nn.Conv2d(200, 200, kernel_size = 3, stride = 2)

        self.conv5a = nn.Conv2d(200, 240, kernel_size = 3)
        self.conv5b = nn.Conv2d(240, 240, kernel_size = 3, stride = 2)

        self.fc1 = nn.Linear(240 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 50)


    def forward(self, x): 
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        
        x = F.relu(self.conv3a(x))
        x = F.relu(self.bn3(self.conv3b(x)))
        
        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        
        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    def loadModel():
        model = NN()
        try:
            model.load_state_dict(torch.load(NN.name))
        except:
            pass
            
        return model