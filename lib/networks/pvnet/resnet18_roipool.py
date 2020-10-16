from torch import nn
import torch
import pickle
from torch.nn import functional as F
import numpy as np
from .resnet import resnet18
from lib.csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer, ransac_voting_layer_v3, estimate_voting_distribution_with_mean
from lib.config import cfg
from .singlestage2 import SimplePnPNet
from .utils import quaternion2rotation


class Resnet18(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(Resnet18, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        self.ver_dim=ver_dim
        self.seg_dim=seg_dim

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(128+fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(64+s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )

        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(raw_dim, seg_dim+ver_dim, 1, 1)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)
        
#         self.onexone = nn.Conv2d(128, 1, 1, 1, 0, bias=False)
        
        self.N_grid = 2
#         self.single_stage = SimplePnPNet(4).cuda()
        self.single_stage = SimplePnPNet(4, 128//self.N_grid*self.N_grid*self.N_grid).cuda()
        self.minNoiseSigma = 0
        self.maxNoiseSigma = 15
        self.minOutlier = 0
        self.maxOutlier = 0.3

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def decode_keypoint(self, output):
        vertex = output['vertex'].permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex.shape
        vertex = vertex.view(b, h, w, vn_2//2, 2)
        mask = torch.argmax(output['seg'], 1)
        if cfg.test.un_pnp:
            mean = ransac_voting_layer_v3(mask, vertex, 512, inlier_thresh=0.99)
            kpt_2d, var = estimate_voting_distribution_with_mean(mask, vertex, mean)
            output.update({'mask': mask, 'kpt_2d': kpt_2d, 'var': var})
        else:
            kpt_2d = ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)
            output.update({'mask': mask, 'kpt_2d': kpt_2d})

    def forward(self, x, seg_gt=torch.zeros(1), feature_alignment=False):       
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)

        fm=self.conv8s(torch.cat([xfc,x8s],1))
#         fm_copy = fm.clone().detach()
        
#         image_features = self.onexone(fm_copy)
#         image_features = image_features.reshape(image_features.shape[0], image_features.shape[2]*image_features.shape[3])

        fm=self.up8sto4s(fm)
        if fm.shape[2]==136:
            fm = nn.functional.interpolate(fm, (135,180), mode='bilinear', align_corners=False)

        fm=self.conv4s(torch.cat([fm,x4s],1))
        fm_copy = fm.clone().detach()
        
        fm=self.up4sto2s(fm)

        fm=self.conv2s(torch.cat([fm,x2s],1))
        fm=self.up2storaw(fm)

        x=self.convraw(torch.cat([fm,x],1))
        seg_pred=x[:,:self.seg_dim,:,:]
        ver_pred=x[:,self.seg_dim:,:,:]
        
        
        #####################################################################
        # single stage model
        #####################################################################
        
        # x refers to horizontal coord and y refers to vertical coord
        # refer to /lib/utils/pvnet/pvnet_data_utils.py 'compute_vertex' function
        pred_dx = []
        pred_dy = []
        pred_x = []
        pred_y = []
        
        ver_pred_reshape = ver_pred.permute(0, 2, 3, 1)
        batch_size, h, w, vn_2 = ver_pred_reshape.shape
        ver_pred_reshape = ver_pred_reshape.reshape(batch_size, h, w, vn_2//2, 2)
        
        if seg_gt.dim()!=1:
            batch_seg_mask = seg_gt
        else:
            batch_seg_mask = torch.argmax(seg_pred, dim=1)
        
        batch_idx_used = []
        image_features = []
        for b in range(batch_size):
            seg_mask = batch_seg_mask[b]
            seg_indices = seg_mask.nonzero()
#             data = {}
#             data['mask_gt'] = seg_gt[b].detach().cpu().numpy()
#             data['mask_pred'] = seg_mask.detach().cpu().numpy()
#             with open('mask.pkl','wb') as fp:
#                 pickle.dump(data, fp)
#             print(seg_indices)
#             input()

    
            # randomly sample 200 points
            if seg_indices.shape[0] >= 200:
                batch_idx_used.append(b)
                
                sampled_indices = np.random.choice(range(seg_indices.shape[0]), 200, replace=False)
                
                dx = ver_pred_reshape[b, seg_indices[sampled_indices,0], seg_indices[sampled_indices,1], :, 0]
                dx = dx.view(-1,1).squeeze()
                dy = ver_pred_reshape[b, seg_indices[sampled_indices,0], seg_indices[sampled_indices,1], :, 1]
                dy = dy.view(-1,1).squeeze()

                pred_dx.append(dx)
                pred_dy.append(dy)
                
                px = seg_indices[sampled_indices,1].float() / w
                px = px.repeat_interleave(vn_2//2)
                py = seg_indices[sampled_indices,0].float() / h
                py = py.repeat_interleave(vn_2//2)
                pred_x.append(px)
                pred_y.append(py)
                
                
                ###################################################
                ## ROI POOL
                ###################################################

                # get bounding box for object
                bbox = np.array([seg_indices[:,0].min(), seg_indices[:,1].min(), seg_indices[:,0].max(), seg_indices[:,1].max()])
                bbox_scaled = (bbox / pow(2,4-self.N_grid)).astype(np.int)
                bbox_scaled[2] = max(bbox_scaled[0]+1, bbox_scaled[2])
                bbox_scaled[3] = max(bbox_scaled[1]+1, bbox_scaled[3])
                # create NXN grid
                N_h = self.N_grid
                N_w = self.N_grid
                bbox_kh = int((bbox_scaled[2] - bbox_scaled[0]) / N_h)
                bbox_kw = int((bbox_scaled[3] - bbox_scaled[1]) / N_w)
                if bbox_kh==0:
                    N_h = 1
                    N_w = self.N_grid*self.N_grid
                    bbox_kh = int((bbox_scaled[2] - bbox_scaled[0]) / N_h)
                    bbox_kw = int((bbox_scaled[3] - bbox_scaled[1]) / N_w)
                elif bbox_kw==0:
                    N_w = 1
                    N_h = self.N_grid*self.N_grid
                    bbox_kh = int((bbox_scaled[2] - bbox_scaled[0]) / N_h)
                    bbox_kw = int((bbox_scaled[3] - bbox_scaled[1]) / N_w)
                # get features and apply max pool on each block
                img_feat = fm_copy[b, :, bbox_scaled[0]:bbox_scaled[2], bbox_scaled[1]:bbox_scaled[3]]
                try:
                    img_feat = nn.MaxPool2d([bbox_kh, bbox_kw], [bbox_kh, bbox_kw])(img_feat)
                except:
                    print(bbox_scaled)
                    print(bbox_kh)
                    print(bbox_kw)
                    input()
                img_feat = img_feat[:, :N_h, :N_w]
                img_feat = img_feat.reshape(img_feat.shape[0]*img_feat.shape[1]*img_feat.shape[2])
                image_features.append(img_feat)

        
        if len(batch_idx_used) != 0:
            pred_dx = torch.stack(pred_dx, 0)
            pred_dy = torch.stack(pred_dy, 0)
            pred_x = torch.stack(pred_x, 0) - 0.5
            pred_y = torch.stack(pred_y, 0) - 0.5
#             if self.training:
#                 # add noise
#                 outlierRatio = np.random.uniform(self.minOutlier, self.maxOutlier)
#                 outlierCnt = int(len(pred_dx) * outlierRatio + 0.5)
#                 outlierChoice = np.random.choice(len(pred_dx), outlierCnt, replace=False)

#                 pred_x[:, outlierChoice] = torch.from_numpy(np.random.uniform(0, 1, size=[pred_x.shape[0], outlierCnt])).float().cuda()
#                 pred_y[:, outlierChoice] = torch.from_numpy(np.random.uniform(0, 1, size=[pred_y.shape[0], outlierCnt])).float().cuda()
#                 # 
#                 pred_dx[:, outlierChoice] = torch.from_numpy(np.random.uniform(-1, 1, size=[pred_dx.shape[0], outlierCnt])).float().cuda()
#                 pred_dy[:, outlierChoice] = torch.from_numpy(np.random.uniform(-1, 1, size=[pred_dy.shape[0], outlierCnt])).float().cuda()
        
#                 noiseSigma = np.random.uniform(self.minNoiseSigma, self.maxNoiseSigma)
#                 noise = np.clip(np.random.normal(0, noiseSigma, pred_dx.shape).astype(np.float32), -0.1*w, 0.1*w)
#                 noise /= w
#                 pred_dx = pred_dx + torch.from_numpy(noise).cuda()
            
#                 noise = np.clip(np.random.normal(0, noiseSigma, pred_dy.shape).astype(np.float32), -0.1*h, 0.1*h)
#                 noise /= h
#                 pred_dy = pred_dy + torch.from_numpy(noise).cuda()
            

            pred_xydxdy = torch.stack([pred_x, pred_y, pred_dx, pred_dy], 2)
            pred_xydxdy = pred_xydxdy.permute(0,2,1)

#             pred_pose = self.single_stage(pred_xydxdy)
#             image_features = image_features[batch_idx_used]
            image_features = torch.stack(image_features, 0)
            pred_pose = self.single_stage(pred_xydxdy, image_features)
        
        batch_pred_pose = torch.zeros(batch_size, 3, 4).cuda()
        mask_pose = torch.zeros(batch_size).cuda()
        
        if len(batch_idx_used) != 0:
            pred_pose_rot = quaternion2rotation(pred_pose[:,:4])
            pred_pose_trans = pred_pose[:,4:].unsqueeze(2)
            pred_pose_rt = torch.cat([pred_pose_rot, pred_pose_trans],2)
            batch_pred_pose[batch_idx_used] = pred_pose_rt
            mask_pose[batch_idx_used] = 1
           

        ret = {'seg': seg_pred, 'vertex': ver_pred, 'pred_pose':batch_pred_pose, 'mask_pose':mask_pose}
#         ret = {'seg': seg_pred, 'vertex': ver_pred}

        if not self.training:
            with torch.no_grad():
                self.decode_keypoint(ret)
        
        return ret


def get_res_pvnet(ver_dim, seg_dim):

    model = Resnet18(ver_dim, seg_dim)
    return model


