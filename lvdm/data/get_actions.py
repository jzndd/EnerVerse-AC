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


def get_actions_from_ee_pose_for_arms(
    gripper, all_ends_p=None, all_ends_o=None, slices=None, delta_act_sidx=None
):
    n = all_ends_p.shape[0]
    slices = list(range(all_ends_p.shape[0]))

    all_left_rpy = []
    all_right_rpy = []
    all_left_quat = []
    all_right_quat = []

    # cam eef 30...CAM_ANGLE...
    # 获取旋转变换对象
    cvt_vis_l = Rotation.from_euler("xyz", np.array(EEF2CamLeft))
    cvt_vis_r = Rotation.from_euler("xyz", np.array(EEF2CamRight))
    for i in slices:
        rot_l = Rotation.from_quat(all_ends_o[i, 0])  # [t = i, 4]
        rot_vis_l = rot_l * cvt_vis_l
        # 左末端到相机坐标系的旋转
        left_vis_quat = np.concatenate((all_ends_p[i, 0], rot_vis_l.as_quat()), axis=0)
        # 3 + 4
        left_rpy = np.concatenate(
            (all_ends_p[i, 0], rot_l.as_euler("xyz", degrees=False)), axis=0
        )
        # 3 + 3
        rot_r = Rotation.from_quat(all_ends_o[i, 1])
        rot_vis_r = rot_r * cvt_vis_r
        right_vis_quat = np.concatenate((all_ends_p[i, 1], rot_vis_r.as_quat()), axis=0)
        right_rpy = np.concatenate(
            (all_ends_p[i, 1], rot_r.as_euler("xyz", degrees=False)), axis=0
        )

        all_left_rpy.append(left_rpy)
        all_right_rpy.append(right_rpy)
        all_left_quat.append(left_vis_quat)
        all_right_quat.append(right_vis_quat)

    # xyz, rpy
    all_left_rpy = np.stack(all_left_rpy)
    all_right_rpy = np.stack(all_right_rpy)
    # xyz, xyzw
    all_left_quat = np.stack(all_left_quat)
    all_right_quat = np.stack(all_right_quat)

    # xyz, xyzw, gripper
    all_abs_actions = np.zeros([n, 16])
    # xyz, rpy, gripper
    all_delta_actions = np.zeros([n - delta_act_sidx, 14])
    # 前4帧重复

    # T -1
    for i in range(0, n):
        all_abs_actions[i, 0:7] = all_left_quat[i, :7]
        all_abs_actions[i, 7] = gripper[slices[i], 0]
        all_abs_actions[i, 8:15] = all_right_quat[i, :7]
        all_abs_actions[i, 15] = gripper[slices[i], 1]
        if i >= delta_act_sidx:
            all_delta_actions[i - delta_act_sidx, 0:6] = (
                all_left_rpy[i, :6] - all_left_rpy[i - 1, :6]
            )
            all_delta_actions[i - delta_act_sidx, 3:6] = normalize_angles(
                all_delta_actions[i - delta_act_sidx, 3:6]
            )
            all_delta_actions[i - delta_act_sidx, 6] = gripper[slices[i], 0] / 120.0
            all_delta_actions[i - delta_act_sidx, 7:13] = (
                all_right_rpy[i, :6] - all_right_rpy[i - 1, :6]
            )
            all_delta_actions[i - delta_act_sidx, 10:13] = normalize_angles(
                all_delta_actions[i - delta_act_sidx, 10:13]
            )
            all_delta_actions[i - delta_act_sidx, 13] = gripper[slices[i], 1] / 120.0

    return all_abs_actions, all_delta_actions


def get_action_from_abs_act(abs_act):
    """
    action_list: [N, 16]
    """
    n_previous = 4
    action, delta_action = get_actions_from_ee_pose_for_arms(
        gripper=np.stack((abs_act[:, 7], abs_act[:, 15]), axis=1),
        all_ends_p=np.stack((abs_act[:, 0:3], abs_act[:, 8:11]), axis=1),
        all_ends_o=np.stack((abs_act[:, 3:7], abs_act[:, 11:15]), axis=1),
        slices=None,
        delta_act_sidx=n_previous,
    )
    action = torch.FloatTensor(action)
    delta_action = torch.FloatTensor(delta_action)
    domain_name = "agibotworld"
    delta_act_meanv, delta_act_stdv = get_action_bias_std(domain_name)
    sep = 3
    delta_action[:, :6] = (delta_action[:, :6] - sep * delta_act_meanv[:, :6]) / (
        sep * delta_act_stdv[:, :6]
    )
    delta_action[:, 7:13] = (delta_action[:, 7:13] - sep * delta_act_meanv[:, 6:]) / (
        sep * delta_act_stdv[:, 6:]
    )
    return action, delta_action