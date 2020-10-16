
import torch
import torch.nn as nn
import numpy as np

class SimplePnPNet(nn.Module):
    def __init__(self, nIn):
        super(SimplePnPNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(nIn, 128, 2, 2)
#         self.conv1 = torch.nn.Conv1d(nIn, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

#         self.fc1 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(1152, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_qt = nn.Linear(256, 7)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        batch_size = x.size(0)
        data_size = x.size(2)  # number of correspondences
        num_pts = data_size // 9

#         import pickle
#         with open('/mbrdi/sqnap1_colomirror/gupansh/local_features_triplet.pkl','wb') as fp:
#             pickle.dump(x, fp)
#         print('saved..')
#         input()
        
        
        ###########################################
        # Triplet loss
        ###########################################
        anchors = x.clone()
        anchors = anchors / anchors.norm(dim=1).unsqueeze(1)
        
        # form positives
        indices = []
        for i in range(9):
            tmp = np.array(range(num_pts))
            np.random.shuffle(tmp)
            indices = np.concatenate([indices, (num_pts*i + tmp)])
        
        positives = x[:,:,indices]
        positives = positives / positives.norm(dim=1).unsqueeze(1)
        
        # form negatives
        indices = []
        for i in range(9):
            tmp = set(range(num_pts * 9)) - set(range(num_pts*i, num_pts*(i+1)))
            tmp = np.array(list(tmp))
            tmp = np.random.choice(tmp, num_pts)
            indices = np.concatenate([indices, tmp])
        
        negatives = x[:,:,indices]        
        negatives = negatives / negatives.norm(dim=1).unsqueeze(1)
        
        # get loss
        S_pos = anchors*positives
        S_pos = S_pos.sum(dim=1)
        S_neg = anchors*negatives
        S_neg = S_neg.sum(dim=1)
        loss = 0.1 - S_pos + S_neg
#         loss = 0.3 - S_pos + S_neg
        tmp = torch.zeros(batch_size, data_size).cuda()
        loss = torch.stack([loss, tmp],2)
        loss = torch.max(loss, dim=2)[0]
        loss = loss.mean(dim=1)
        

        x = x.permute(0, 2, 1)
        x = x.view(batch_size, 9, -1, 128)
#         x = torch.max(x, dim=2, keepdim=True)[0]
        x = torch.mean(x, dim=2, keepdim=True)
#         x = torch.median(x, dim=2, keepdim=True)[0]
#         x = x.view(batch_size, 128, -1, 8)
#         x = torch.max(x, dim=2, keepdim=True)[0]

        x = x.view(batch_size, 1152)
#         x = x.view(batch_size, 1024)
        # 
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        # 
        qt = self.fc_qt(x)
        return qt, loss