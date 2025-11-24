import numpy as np
import os
import h5py
from scipy.spatial.transform import Rotation
from lvdm.data.traj_vis_statistics import EEF2CamLeft, EEF2CamRight
import torch
from lvdm.data.statistics import StatisticInfo

def normalize_angles(radius):
    radius_normed = np.mod(radius, 2 * np.pi) - 2 * np.pi * (np.mod(radius, 2 * np.pi) > np.pi)
    return radius_normed

def get_action_bias_std(domain_name):
    return torch.tensor(StatisticInfo[domain_name]['mean']).unsqueeze(0), torch.tensor(StatisticInfo[domain_name]['std']).unsqueeze(0)

def get_actions_from_ee_pose(gripper, all_ends_p=None, all_ends_o=None, slices=None, delta_act_sidx=None):

    if delta_act_sidx is None:
        delta_act_sidx = 1
    # delta_act_sidx = 4
    if slices is None:
        ### the first frame is repeated to fill memory
        n = all_ends_p.shape[0]-1+delta_act_sidx
        # n = T + 3
        slices = [0,]*(delta_act_sidx-1) + list(range(all_ends_p.shape[0]))
        # 0 * 3 + [0, 1, 2, ..., T-1]
    else:
        n = len(slices)
    #print(f'slices:{slices}')
    all_rpy = []
    all_quat = []

    for i in slices:
        # 0 * 3 + [0, 1, 2, ..., T-1]
        quat_wxyz = all_ends_o[i]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rot = Rotation.from_quat(quat_xyzw)

        xyz_quat = np.concatenate((all_ends_p[i], rot.as_quat()), axis=0)
        # 3 + 4
        xyz_rpy = np.concatenate((all_ends_p[i], rot.as_euler("xyz", degrees=False)), axis=0)
        # 3 + 3

        all_rpy.append(xyz_rpy)
        all_quat.append(xyz_quat)
    ### xyz, rpy
    all_rpy = np.stack(all_rpy)
    ### xyz, xyzw
    all_quat = np.stack(all_quat)

    ### xyz, xyzw, gripper
    all_abs_actions = np.zeros([n, 8])
    ### xyz, rpy, gripper
    all_delta_actions = np.zeros([n-delta_act_sidx, 7])
    # 前4帧重复
    # T -1 
    for i in range(0, n):
        all_abs_actions[i, 0:7] = all_quat[i, :7]
        all_abs_actions[i, 7] = gripper[slices[i]]
        if i >= delta_act_sidx:
            all_delta_actions[i-delta_act_sidx, 0:6] = all_rpy[i, :6] - all_rpy[i-1, :6]
            all_delta_actions[i-delta_act_sidx, 3:6] = normalize_angles(all_delta_actions[i-delta_act_sidx, 3:6])
            all_delta_actions[i-delta_act_sidx, 6] = gripper[slices[i]] / 1.0
    #print(f'end get_actions_maniskill')
    return all_abs_actions, all_delta_actions