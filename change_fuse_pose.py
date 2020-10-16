import numpy as np
import os
import glob
import pickle


PATH = '/pfs/rdi/cei/algo_train/gupansh/clean-pvnet/data/linemod/fuse_new2/'
linemod_cls_names = ['ape', 'cam', 'cat', 'duck', 'glue', 'iron', 'phone', 'benchvise', 'can', 'driller', 'eggbox', 'holepuncher', 'lamp']
linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])
linemod_K_inv = np.linalg.inv(linemod_K)

ann_num = len(glob.glob(os.path.join(PATH, '*.pkl')))
for ind in range(ann_num):
    print(ind)
    
    info_path = os.path.join(PATH, '{}_info.pkl'.format(ind))
    if not os.path.exists(info_path):
        print(ind, ' does not exist...')
        input()
        continue
    
    with open(info_path, 'rb') as fp:
        begins, poses = pickle.load(fp)
    
    poses_new = np.zeros(poses.shape)
    begins_new = np.zeros(begins.shape)
    for cls_idx in range(len(linemod_cls_names)):
        
        pose = poses[cls_idx]
        beg = begins[cls_idx]
        
        K = linemod_K.copy()
        K[0, 2] += beg[1]
        K[1, 2] += beg[0]
        
        npose = np.dot(linemod_K_inv, np.dot(K, pose))
        poses_new[cls_idx] = npose
    
    with open(os.path.join(PATH, '{}_info.pkl'.format(ind)), 'wb') as fp:
        pickle.dump([begins_new, poses_new], fp)
        
        
        
    