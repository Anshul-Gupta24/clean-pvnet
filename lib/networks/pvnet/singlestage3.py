
import torch
import torch.nn as nn

class SimplePnPNet(nn.Module):
    def __init__(self, nIn):
        super(SimplePnPNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(nIn, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

        self.fc1_rot = nn.Linear(1152, 512)
        self.fc2_rot = nn.Linear(512, 256)
        self.fc_rot = nn.Linear(256, 4)
        self.fc1_tr = nn.Linear(1152, 512)
        self.fc2_tr = nn.Linear(512, 256)
        self.fc_tr = nn.Linear(256, 3)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        batch_size = x.size(0)
        data_size = x.size(2)  # number of correspondences

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        x = x.permute(0, 2, 1)
        x = x.view(batch_size, -1, 9, 128)
        x = torch.max(x, dim=1, keepdim=True)[0]
#         x = x.view(batch_size, 128, -1, 8)
#         x = torch.max(x, dim=2, keepdim=True)[0]
        # x = torch.mean(x, dim=2, keepdim=True)

        x = x.view(batch_size, 1152)
        # 
        x_rot = self.act(self.fc1_rot(x))
        x_rot = self.act(self.fc2_rot(x_rot))
        x_rot = self.fc_rot(x_rot)
        
        x_tr = self.act(self.fc1_tr(x))
        x_tr = self.act(self.fc2_tr(x_tr))
        x_tr = self.fc_tr(x_tr)
        # 
        
        qt = torch.cat([x_rot, x_tr], 1)

        return qt