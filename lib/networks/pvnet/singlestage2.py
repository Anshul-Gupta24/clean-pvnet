
import torch
import torch.nn as nn

class SimplePnPNet(nn.Module):
    def __init__(self, nIn, img_feat_size):
        super(SimplePnPNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(nIn, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

        self.fc1 = nn.Linear(1152+img_feat_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_qt = nn.Linear(256, 7)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, image_features):
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
        x = torch.cat([x, image_features], 1)
        # 
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        # 
        qt = self.fc_qt(x)
        return qt