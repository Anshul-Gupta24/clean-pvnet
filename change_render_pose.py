import numpy as np
import os
import glob
import pickle


PATH = '/pfs/rdi/cei/algo_train/gupansh/clean-pvnet/data/linemod/renders_new/holepuncher/'
linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])
linemod_K_inv = np.linalg.inv(linemod_K)

K = np.array([[700., 0., 320.],
                    [0., 700., 240.],
                    [0., 0., 1.]])

ann_num = len(glob.glob(os.path.join(PATH, '*.pkl')))
print(ann_num)
input()
for ind in range(ann_num):
    print(ind)
    
    with open(os.path.join(PATH, '{}_RT.pkl'.format(ind)), 'rb') as fp:
        poses = pickle.load(fp)
    
    pose = poses['RT']
    npose = np.dot(linemod_K_inv, np.dot(K, pose))
    poses_new = {}        
    poses_new['RT'] = npose
    poses_new['K'] = linemod_K
    
    with open(os.path.join(PATH, '{}_RT.pkl'.format(ind)), 'wb') as fp:
        pickle.dump(poses_new, fp)
        
        
        
    