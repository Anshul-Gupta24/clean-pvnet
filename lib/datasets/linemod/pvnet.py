import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
import pickle
import os
from PIL import Image
from scipy.spatial.transform import Rotation as R
from lib.utils.pvnet import pvnet_data_utils, pvnet_linemod_utils, visualize_utils
from lib.utils.linemod import linemod_config
from lib.datasets.augmentation import *
import random
import torch
from lib.config import cfg
from lib.utils.linemod.linemod_config import linemod_K

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

class Dataset(data.Dataset):

    def __init__(self, ann_file, data_root, split, transforms=None):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms
        self.cfg = cfg
        
        self.cls = cfg.cls_type

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        path = self.coco.loadImgs(int(img_id))[0]['file_name']

        pose = np.array(anno['pose'])
#         K = anno['K']
        K = linemod_K
        inp = Image.open(path)
        kpt_2d = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        
        # for symmetric classes
#         kpt_2d = np.concatenate([anno['corner_2d'], [anno['center_2d']]], axis=0)
#         kpt_3d = np.concatenate([anno['corner_3d'], [anno['center_3d']]], axis=0)

        cls_idx = linemod_config.linemod_cls_names.index(anno['cls']) + 1
        mask = pvnet_data_utils.read_linemod_mask(anno['mask_path'], anno['type'], cls_idx)
        
        return inp, kpt_2d, kpt_3d, mask, pose, K
    
    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        height, width = 480, 640
        img_id = self.img_ids[index]

        img, kpt_2d, kpt_3d, mask, pose, K = self.read_data(img_id)

        if self.split == 'train':
#             inp = np.asarray(img).astype(np.uint8)
            inp, kpt_2d, mask, pose = self.augment(img, mask, kpt_2d, pose, K, height, width)
        else:
            inp = img
        
        K = np.array(K)
        dist_coeffs = np.zeros((4,1))
        (_, rvec, tvec) = cv2.solvePnP(kpt_3d, kpt_2d.astype(np.float32), K, dist_coeffs)
        rot = cv2.Rodrigues(rvec)[0]
        pnp_pose = np.concatenate([rot, tvec], axis=1)
        pose_old = pnp_pose.copy()

        # for symmetric objects
        if self.cls in ['eggbox', 'glue']:
            rot = R.from_matrix(pnp_pose[:3,:3])
            euler_angles = rot.as_euler('xyz', degrees=True)
            if euler_angles[2] >= 0:
                #corrected pose
                pnp_pose[:3,:3] = np.dot(pnp_pose[:3,:3], np.array([[-1,0,0],[0,-1,0],[0,0,1]]))
                # correct keypoints
                kpt_2d = project(kpt_3d, K, pnp_pose)
                
#         print('computed pose', pose)
#         print('old pose', pose_old)
#         print('pnp pose ', pnp_pose)
#         data = {}
#         data['img'] = inp
#         data['img_old'] = img
#         data['mask'] = mask
#         data['pose_old'] = pose_old
#         data['pose'] = pnp_pose
#         data['kpt_2d'] = kpt_2d
#         data['K'] = K
#         with open('/mbrdi/sqnap1_colomirror/gupansh/data.pkl','wb') as fp:
#             pickle.dump(data, fp)
#         print('saved..')
#         input()
        
        
        if self._transforms is not None:
            inp, kpt_2d, mask = self._transforms(inp, kpt_2d, mask)
        
        # shape is (18, H, W) y1,x1,y2,x2 ...y9,x9 
        vertex = pvnet_data_utils.compute_vertex(mask, kpt_2d).transpose(2, 0, 1)
        
        ret = {'inp': inp, 'mask': mask.astype(np.uint8), 'vertex': vertex, 'img_id': img_id, 'pose':pnp_pose, 'pose_pnp':pnp_pose, 'K': K, 'xy': kpt_2d, 'meta': {}}
        # visualize_utils.visualize_linemod_ann(torch.tensor(inp), kpt_2d, mask, True)

        return ret

    def __len__(self):
        return len(self.img_ids)

    def augment(self, img, mask, kpt_2d, pose, K, height, width):
       
        # add one column to kpt_2d for convenience to calculate
        hcoords = np.concatenate((kpt_2d, np.ones((9, 1))), axis=-1)
        img = np.asarray(img).astype(np.uint8)
        foreground = np.sum(mask)
        
        # randomly mask out to add occlusion
#         if foreground > 0:
#             img, mask, hcoords = rotate_instance(img, mask, hcoords, self.cfg.train.rotate_min, self.cfg.train.rotate_max)
#             img, mask, hcoords = crop_resize_instance_v1(img, mask, hcoords, height, width,
#                                                          self.cfg.train.overlap_ratio,
#                                                          self.cfg.train.resize_ratio_min,
#                                                          self.cfg.train.resize_ratio_max)
#         else:
#             img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)
        
    
############################################################
        # New augmentation for pose
############################################################
        if foreground > 0:
            mask1 = mask.copy()
#             img, mask, hcoords, pose = translate_instance_pose(img, mask, hcoords, pose, K)
            img, mask, hcoords, pose = crop_resize_instance_v1_pose(img, mask, hcoords, pose, K, height, width,
                                                         self.cfg.train.overlap_ratio,
                                                         self.cfg.train.resize_ratio_min,
                                                         self.cfg.train.resize_ratio_max)
            mask2 = mask.copy()
            img, mask, hcoords, pose = rotate_instance_pose(img, mask, hcoords, pose, K, self.cfg.train.rotate_min, self.cfg.train.rotate_max)
            mask3_sum = mask.sum()
            try:
                img, mask = mask_out_instance(img, mask)
            except:
                data = {}
                data['img'] = img
                data['mask1'] = mask1
                data['mask2'] = mask2
                data['pose'] = pose
                with open('data.pkl','wb') as fp:
                    pickle.dump(data, fp)
                print('saved..')
#                 input()
            
        kpt_2d = hcoords[:, :2]

        return img, kpt_2d, mask, pose
