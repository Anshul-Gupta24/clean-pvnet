from torch import nn
import torch
import pickle
from torch.nn import functional as F
import numpy as np
from .resnet import resnet18
from lib.csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer, ransac_voting_layer_v3, estimate_voting_distribution_with_mean
from lib.config import cfg
from .singlestage_triplet import SimplePnPNet
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
        self.single_stage = SimplePnPNet(4).cuda()
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
        encoder = [x2s, x4s, x8s, x16s, x32s, xfc]
        decoder = []

        fm=self.conv8s(torch.cat([xfc,x8s],1))
        decoder.append(fm.clone())

        fm=self.up8sto4s(fm)
        if fm.shape[2]==136:
            fm = nn.functional.interpolate(fm, (135,180), mode='bilinear', align_corners=False)

        fm=self.conv4s(torch.cat([fm,x4s],1))
        decoder.append(fm.clone())
        
        fm=self.up4sto2s(fm)
        fm=self.conv2s(torch.cat([fm,x2s],1))
        decoder.append(fm.clone())
        
        fm=self.up2storaw(fm)
        x=self.convraw(torch.cat([fm,x],1))
        
        # save encoder, decoder features
#         encoder = [f.detach().cpu().numpy() for f in encoder]
#         decoder = [f.detach().cpu().numpy() for f in decoder]
#         with open('/mbrdi/sqnap1_colomirror/gupansh/encoder_holepuncher.pkl','wb') as fp:
#             pickle.dump(encoder, fp)
#         with open('/mbrdi/sqnap1_colomirror/gupansh/decoder_holepuncher.pkl','wb') as fp:
#             pickle.dump(decoder, fp)
#         input()

        
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
            conf, batch_seg_mask = torch.max(seg_pred, dim=1)
        
        # save segmentation, confidence
#         with open('/mbrdi/sqnap1_colomirror/gupansh/segmentation.pkl','wb') as fp:
#             pickle.dump(batch_seg_mask.detach().cpu().numpy(), fp)
#         with open('/mbrdi/sqnap1_colomirror/gupansh/confidence.pkl','wb') as fp:
#             pickle.dump(conf.detach().cpu().numpy(), fp)
#         input()
        
        
        batch_idx_used = []
        image_features = []
        triplet_loss = torch.zeros(batch_size).cuda()
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
                dx = dx.transpose(0,1).contiguous()
                dx = dx.view(-1,1).squeeze()
                dy = ver_pred_reshape[b, seg_indices[sampled_indices,0], seg_indices[sampled_indices,1], :, 1]
                dy = dy.transpose(0,1).contiguous()
                dy = dy.view(-1,1).squeeze()

                pred_dx.append(dx)
                pred_dy.append(dy)
                
                px = seg_indices[sampled_indices,1].float() / w
                px = px.repeat(vn_2//2)
                py = seg_indices[sampled_indices,0].float() / h
                py = py.repeat(vn_2//2)
                pred_x.append(px)
                pred_y.append(py)
                
        
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

            pred_pose, triplet_loss = self.single_stage(pred_xydxdy)
        
        batch_pred_pose = torch.zeros(batch_size, 3, 4).cuda()
        mask_pose = torch.zeros(batch_size).cuda()
        
        if len(batch_idx_used) != 0:
            pred_pose_rot = quaternion2rotation(pred_pose[:,:4])
            pred_pose_trans = pred_pose[:,4:].unsqueeze(2)
            pred_pose_rt = torch.cat([pred_pose_rot, pred_pose_trans],2)
            batch_pred_pose[batch_idx_used] = pred_pose_rt
            mask_pose[batch_idx_used] = 1
           

        ret = {'seg': seg_pred, 'vertex': ver_pred, 'pred_pose':batch_pred_pose, 'mask_pose':mask_pose, 'triplet_loss': triplet_loss}
#         ret = {'seg': seg_pred, 'vertex': ver_pred}

        if not self.training:
            with torch.no_grad():
                self.decode_keypoint(ret)
        
        return ret


def get_res_pvnet(ver_dim, seg_dim):

    model = Resnet18(ver_dim, seg_dim)
    return model


