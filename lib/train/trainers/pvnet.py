import torch.nn as nn
from lib.utils import net_utils
import torch
import numpy as np
import pickle
from lib.config import cfg

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        cls = cfg.cls_type

        self.vote_crit = torch.nn.functional.smooth_l1_loss
        self.seg_crit = nn.CrossEntropyLoss()
        
        with open('data/linemod/'+cls+'/translation_min.pkl', 'rb') as fp:
            self.translation_min = pickle.load(fp)
        self.translation_min = torch.FloatTensor(self.translation_min)
        with open('data/linemod/'+cls+'/translation_max.pkl', 'rb') as fp:
            self.translation_max = pickle.load(fp)
        self.translation_max = torch.FloatTensor(self.translation_max)

    def forward(self, batch):    
        
        output = self.net(batch['inp'])

        scalar_stats = {}
        loss = 0

        if 'pose_test' in batch['meta'].keys():
            loss = torch.tensor(0).to(batch['inp'].device)
            return output, loss, {}, {}

        weight = batch['mask'][:, None].float()
        vote_loss = self.vote_crit(output['vertex'] * weight, batch['vertex'] * weight, reduction='sum')
        vote_loss = vote_loss / weight.sum() / batch['vertex'].size(1)
        
#         #####################
#         # vote loss symmetric
#         #####################
#         vote_loss1 = self.vote_crit(output['vertex'] * weight, batch['vertex'] * weight, reduction='sum')
#         vote_loss1 = vote_loss1 / weight.sum() / batch['vertex'].size(1)
#         idxs = [6, 7, 4, 5, 2, 3, 0, 1, 8]
#         bv = batch['vertex']
#         batch_size, vn_2, h, w = bv.shape
#         bv = bv.view(batch_size, vn_2//2, 2, h, w)
#         bv = bv[:,idxs,:,:,:]
#         bv = bv.view(batch_size, vn_2, h, w)
#         vote_loss2 = self.vote_crit(output['vertex'] * weight, bv * weight, reduction='sum')
#         vote_loss2 = vote_loss2 / weight.sum() / batch['vertex'].size(1)
#         vote_loss = min(vote_loss1, vote_loss2)
        
        scalar_stats.update({'vote_loss': vote_loss})
        loss += vote_loss

        mask = batch['mask'].long()
        seg_loss = self.seg_crit(output['seg'], mask)
        scalar_stats.update({'seg_loss': seg_loss})
        loss += seg_loss
        
        mask_pose = output['mask_pose']
#         ##############################################
#         # xy loss
#         ##############################################
#         gt_xy = batch['xy'].float()
#         gt_xy = gt_xy.repeat(1,200,1)
#         pred_xy = output['pred_xy']
#         xy_loss = torch.norm(gt_xy - pred_xy, dim=2)
#         alpha = 0.0001
#         xy_loss = alpha*xy_loss.mean()
#         scalar_stats.update({'xy_loss': xy_loss})
#         loss += alpha*xy_loss
        
        
        ##############################################
        # pose loss
        ##############################################
        pose_targets = batch['pose'].float()
        pose_targets_pnp = batch['pose_pnp'].float()
        pose_pred = output['pred_pose'].clone()
        intrinsic = batch['K'].float()
             
        # normalize translation
#         means = torch.tensor([ 5.0421385e-04, -5.2776873e-02,  8.8903457e-01]).cuda()
#         vars = torch.tensor([0.00630739, 0.00344274, 0.01582594]).cuda() * 2
#         stds = torch.sqrt(vars)
#         pose_targets[:,:,3] -= means
#         pose_targets[:,:,3] /= vars
        minT = self.translation_min.view(-1,3).cuda()
        maxT = self.translation_max.view(-1,3).cuda()
        pose_pred[:,:,3] =  (pose_pred[:,:,3] + 0.5) * (maxT - minT) + minT
        
#         # increase weight of translation
        pose_pred[:,:2,3] *= 100    
        pose_targets[:,:2,3] *= 100
        pose_pred[:,2,3] *= 10 
        pose_targets[:,2,3] *= 10
        
        batch_size = pose_targets.shape[0]
        
        # 3d points are unit cube at origin
        point_3d = 0.5 * torch.from_numpy(np.array([1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1])).float()
#         point_3d = 0.05 * torch.from_numpy(np.array([1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1])).float()
        point_3d = point_3d.view(-1, 3)
        # cat 3D points
#         point_3d = 10 * torch.from_numpy(np.array([[0.021784409880638123, -0.05869993939995766, -0.041461799293756485], [-0.020935669541358948, 0.065584197640419, 0.03955494984984398], [0.027798950672149658, 0.02978627011179924, -0.04446335881948471], [0.0013946900144219398, -0.05708838999271393, 0.03459186106920242], [0.01266971044242382, 0.015784140676259995, 0.07163164764642715], [-0.03255600109696388, -0.03509996086359024, -0.0401097796857357], [-0.029417019337415695, 0.022817589342594147, -0.03967477008700371], [0.02389688976109028, 0.04313550889492035, 0.03052332065999508]])).float()
        point_3d = point_3d.repeat(batch_size,1).view(batch_size,8,3).cuda()
        
        ###############################
        # 3d error
        ###############################
        model_pred = torch.bmm(pose_pred[:, :, :3], point_3d.transpose(1, 2)) + pose_pred[:, :, 3].unsqueeze(dim=2)
        model_targets = torch.bmm(pose_targets[:, :, :3], point_3d.transpose(1, 2)) + pose_targets[:, :, 3].unsqueeze(dim=2)
        
        pose_loss = (model_pred - model_targets).norm(dim=1).mean(dim=1)

        
#         #####################
#         # symmetric pose loss
#         #####################
#         pose_loss = torch.zeros(batch_size).cuda()
#         for b in range(batch_size):
#             mp = model_pred[b].transpose(0,1)
#             mt = model_targets[b].transpose(0,1)
#             idxs = [6, 7, 4, 5, 2, 3, 0, 1]
#             pose_loss[b] = min((mp - mt).norm(dim=1).mean(), (mp[idxs] - mt).norm(dim=1).mean())

# #         pose_loss = pose_loss * mask_pose
# #         if mask_pose.sum() > 0:
# #             pose_loss = pose_loss.sum() / mask_pose.sum()
# #         else:
# #             pose_loss = mask_pose.sum()
        
        ###############################
        # 2d error
        ###############################
#         model_pred_2d = torch.bmm(intrinsic, model_pred)
#         model_targets_2d = torch.bmm(intrinsic, model_targets)
        
# #         valid_indices = mask_pose.nonzero().squeeze()
# #         if valid_indices.nelement()!=0:            
# #             model_pred_2d = model_pred_2d[valid_indices]
# #             model_targets_2d = model_targets_2d[valid_indices]
# #             if valid_indices.nelement()==1:
# #                 model_pred_2d = model_pred_2d.unsqueeze(0)
# #                 model_targets_2d = model_targets_2d.unsqueeze(0)

#         model_pred_2dx = model_pred_2d[:,0,:] / model_pred_2d[:,2,:]
#         model_pred_2dy = model_pred_2d[:,1,:] / model_pred_2d[:,2,:]
#         model_targets_2dx = model_targets_2d[:,0,:] / model_targets_2d[:,2,:]
#         model_targets_2dy = model_targets_2d[:,1,:] / model_targets_2d[:,2,:]

#         model_pred_2d = torch.cat((model_pred_2dx.unsqueeze(1), model_pred_2dy.unsqueeze(1)), dim=1)
#         model_targets_2d = torch.cat((model_targets_2dx.unsqueeze(1), model_targets_2dy.unsqueeze(1)), dim=1)
        
#         pose_loss = (model_pred_2d - model_targets_2d).norm(dim=1).mean(dim=1)
# #         else:
# #             pose_loss = torch.zeros(1).cuda()
        
        alpha = 0.01    #scaling factor for pose loss
        pose_loss = alpha*pose_loss.mean()
        loss += pose_loss
        scalar_stats.update({'pose_loss': pose_loss})
        
        
        ######################################################
        # triplet loss
        ######################################################
        triplet_loss = output['triplet_loss']
        alpha = 0.1     # for end-to-end training
#         alpha = 0.05    # for only mlp training
        triplet_loss = alpha*triplet_loss.mean()
        loss += triplet_loss
        scalar_stats.update({'triplet_loss': triplet_loss})

        
        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
