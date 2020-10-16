from torch import nn
import torch
import pickle
from torch.nn import functional as F
import numpy as np
from .resnet import resnet18
from lib.csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer, ransac_voting_layer_v3, estimate_voting_distribution_with_mean
from lib.config import cfg
from .singlestage_triplet import SimplePnPNet
# from .singlestage import SimplePnPNet
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
        pred_x = []
        pred_y = []
        pred_dx = []
        pred_dy = []
        
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
            if seg_indices.shape[0] >= 250:
                batch_idx_used.append(b)
                
                sampled_indices_total = np.random.choice(range(seg_indices.shape[0]), 250, replace=False)
                sampled_indices = sampled_indices_total[:200]
                
                dx = ver_pred_reshape[b, seg_indices[sampled_indices,0], seg_indices[sampled_indices,1], :, 0]
                dx = dx.transpose(0,1).contiguous()
                dx = dx.view(-1,1).squeeze()
                dy = ver_pred_reshape[b, seg_indices[sampled_indices,0], seg_indices[sampled_indices,1], :, 1]
                dy = dy.transpose(0,1).contiguous()
                dy = dy.view(-1,1).squeeze()
                dxy = torch.stack([dx, dy], 1)
                dxy_norm = torch.norm(dxy,dim=1)
                dx = dx / dxy_norm
                dy = dy / dxy_norm
                
                px = seg_indices[sampled_indices,1].float()
                px = px.repeat(vn_2//2)
#                 px = px.repeat_interleave(vn_2//2)
                py = seg_indices[sampled_indices,0].float()
                py = py.repeat(vn_2//2)
#                 py = py.repeat_interleave(vn_2//2)
                
                sampled_indices = sampled_indices_total[200:]
                sampled_indices = sampled_indices.repeat(4)
                dx_pair = ver_pred_reshape[b, seg_indices[sampled_indices,0], seg_indices[sampled_indices,1], :, 0]
                dx_pair = dx_pair.transpose(0,1).contiguous()
                dx_pair = dx_pair.view(-1,1).squeeze()
                dy_pair = ver_pred_reshape[b, seg_indices[sampled_indices,0], seg_indices[sampled_indices,1], :, 1]
                dy_pair = dy_pair.transpose(0,1).contiguous()
                dy_pair = dy_pair.view(-1,1).squeeze()
                dxy_pair = torch.stack([dx_pair, dy_pair], 1)
                dxy_pair_norm = torch.norm(dxy_pair,dim=1)
                dx_pair = dx_pair / dxy_pair_norm
                dy_pair = dy_pair / dxy_pair_norm
                
                px_pair = seg_indices[sampled_indices,1].float()
                px_pair = px_pair.repeat(vn_2//2)
#                 px_pair = px_pair.repeat_interleave(vn_2//2)
                py_pair = seg_indices[sampled_indices,0].float()
                py_pair = py_pair.repeat(vn_2//2)
#                 py_pair = py_pair.repeat_interleave(vn_2//2)
                
                A = torch.stack([dx_pair, -dx, dy_pair, -dy], 1)
                A = A.reshape(1800, 2, 2)
                
                B = torch.stack([px - px_pair, py - py_pair], 1)
                B = B.unsqueeze(2)
                
                X, _ = torch.solve(B, A)
                X_1 = X[:,1].squeeze()
#                 dx.retain_grad()
#                 print('dx', dx.grad)
#                 dx.register_hook(lambda x: print('dx', x))
#                 X_1.retain_grad()
#                 print('X_1', X_1.grad)
#                 X_1.register_hook(lambda x: print('X_1', x))
#                 input()
                delx = X_1*dx
                dely = X_1*dy 
                
                pred_x.append(px / w)
                pred_y.append(py / h)
                pred_dx.append(delx / w)
                pred_dy.append(dely / w)
#                 print('px', px.view(200,9)[:5] + delx.view(200,9)[:5])
#                 print('py', py.view(200,9)[:5] + dely.view(200,9)[:5])
#                 input()
                
        
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

            pred_pose, triplet_loss = self.single_stage(pred_xydxdy.detach())
#             pred_pose = self.single_stage(pred_xydxdy.detach())
        
        batch_pred_pose = torch.zeros(batch_size, 3, 4).cuda()
#         batch_pred_xy = torch.zeros(batch_size, 1800, 2).cuda()
        mask_pose = torch.zeros(batch_size).cuda()
        
        if len(batch_idx_used) != 0:
            pred_pose_rot = quaternion2rotation(pred_pose[:,:4])
            pred_pose_trans = pred_pose[:,4:].unsqueeze(2)
            pred_pose_rt = torch.cat([pred_pose_rot, pred_pose_trans],2)
            batch_pred_pose[batch_idx_used] = pred_pose_rt
            mask_pose[batch_idx_used] = 1
#             batch_pred_xy[batch_idx_used] = pred_xy.permute(0,2,1)
           

        ret = {'seg': seg_pred, 'vertex': ver_pred, 'pred_pose':batch_pred_pose, 'mask_pose':mask_pose, 'triplet_loss': triplet_loss}
#         ret = {'seg': seg_pred, 'vertex': ver_pred, 'pred_pose':batch_pred_pose, 'mask_pose':mask_pose, 'pred_xy': batch_pred_xy}
#         ret = {'seg': seg_pred, 'vertex': ver_pred}

        if not self.training:
            with torch.no_grad():
                self.decode_keypoint(ret)
        
        return ret


def get_res_pvnet(ver_dim, seg_dim):

    model = Resnet18(ver_dim, seg_dim)
    return model


