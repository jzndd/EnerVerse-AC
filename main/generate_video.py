import sys
import os
import torch
import numpy as np
import math
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import json
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from lvdm.data.get_actions import get_actions_from_ee_pose, get_action_bias_std
from evac_utils.general_utils import load_checkpoints, instantiate_from_config
from lvdm.data.statistics import StatisticInfo
from scipy.spatial.transform import Rotation as R

def get_action_from_npy_for_eval(
        npy_path, n_chunk, chunk, n_previous, sep=1, domain_name="agibotworld"
    ):
    if n_chunk > 0:
        slices = list(range(0, n_chunk*chunk+n_previous))
    else:
        slices = None
    action, delta_action = prepare_action_from_npy_for_eval(npy_path, slices=slices, delta_act_sidx=n_previous)
    action = torch.FloatTensor(action)
    delta_action = torch.FloatTensor(delta_action)

    action = torch.cat([action, action], dim=1)
    delta_action = torch.cat([delta_action, delta_action], dim=1)
    delta_act_meanv, delta_act_stdv = get_action_bias_std('agibotworld')
    delta_action[:, :6] = (delta_action[:, :6] - sep*delta_act_meanv[:, :6]) / (sep*delta_act_stdv[:, :6])
    delta_action[:, 7:13] = (delta_action[:, 7:13] - sep*delta_act_meanv[:, 6:]) / (sep*delta_act_stdv[:, 6:])

    return action, delta_action

def prepare_action_from_npy_for_eval(npy_file, slices=None, delta_act_sidx=1):
    """
    read and parse .npy file, and obtain the absolute actions and the action differences
    """
    action = np.load(npy_file, allow_pickle=True)[:,0]
    action = action.astype(np.float32)
    all_ends_p = action[:, 0:3]        # 位置 [T, 3]
    all_ends_o = action[:, 3:7]        # 四元数 [T, 4]
    all_abs_gripper = action[:, 7:8]   # gripper [T, 1]
    all_abs_actions, all_delta_actions = get_actions_from_ee_pose(
        gripper=all_abs_gripper,
        slices=slices,
        delta_act_sidx=delta_act_sidx,
        all_ends_p=all_ends_p,
        all_ends_o=all_ends_o,
    )
    return all_abs_actions, all_delta_actions

def load_model(config):
    model = instantiate_from_config(config.model)
    model = load_checkpoints(model, config.model, ignore_mismatched_sizes=False)
    return model


def load_config(args):
    config_file = args.config_path
    config = OmegaConf.load(config_file)
    config.model.pretrained_checkpoint = args.ckp_path
    return config


def get_image(npy_path, n):
    arr = np.load(npy_path, allow_pickle=True)[:,0]
    img = arr[0]  # 只取第一帧的前3通道，shape: [3, h, w]
    img = torch.from_numpy(img).float().unsqueeze(1) / 255.0  # [3, 1, h, w]
    img = img.repeat(1, n, 1, 1)  # [3, n, h, w]
    return img



def get_caminfo(n):
    ###========== 固定内参 ==========
    intrinsic = torch.tensor([
        [309.02, 0, 128],
        [0, 309.02, 128],
        [0, 0, 1]
    ], dtype=torch.float)

    # ========== 固定外参 ==========
    pos = np.array([0.6586131746834771, 0.0, 1.6103500240372423])
    quat_wxyz = np.array([0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349])
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # xyzw
    rot = R.from_quat(quat_xyzw).as_matrix()  # 3x3

    c2w = torch.eye(4, dtype=torch.float)
    c2w[:3, :3] = torch.from_numpy(rot).float()
    c2w[:3, 3] = torch.from_numpy(pos).float()
    # c2w = torch.tensor([
    #     [-7.26818450e-08,  6.28266450e-01, -7.77998244e-01,  6.58613175e-01],
    #     [ 1.00000000e+00, -7.26818450e-08, -1.52115266e-07,  0.00000000e+00],
    #     [-1.52115266e-07, -7.77998244e-01, -6.28266450e-01,  1.61035002e+00],
    #     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    # ], dtype=torch.float)
    w2c = torch.linalg.inv(c2w)

    # 所有时间步都一样
    T = n
    c2ws = c2w.unsqueeze(0).repeat(T, 1, 1)
    w2cs = w2c.unsqueeze(0).repeat(T, 1, 1)


    return c2ws, w2cs, intrinsic

def main(args):

    seed_everything(args.seed)
    device = torch.device(args.device)

    ### load config
    config = load_config(args)

    chunk = config.chunk
    n_previous = config.n_previous

    ### 
    img = get_image(
        args.video_path, n_previous
    ) # img.shape: [channel, n_previous, H, W]
    #img shape: torch.Size([3, 4, 480, 640])
    ###
    action, delta_action = get_action_from_npy_for_eval(
        args.input_path, args.n_chunk, chunk, n_previous,
        sep=config.data.params.train.params.max_sep,
        domain_name="agibotworld"
    )
    #input('--- IGNORE ---')
    n = action.shape[0]
    #input('--- IGNORE ---')
    ###

    c2w, w2c, intrinsic = get_caminfo(n)
    ##c2w shape: torch.Size([337, 4, 4]), w2c shape: torch.Size([337, 4, 4]), intrinsic shape: torch.Size([3, 3])
    ###
    model = load_model(config).to(device=device)
    model.eval()

    num_chunk = min(8, int(math.ceil((float(n)-n_previous)/chunk)))

    with torch.cuda.amp.autocast(dtype=torch.float32):
        model.inference(
            config, img, action, delta_action,
            c2w, w2c, intrinsic,
            args.save_root, num_chunk,
            chunk=chunk, n_previous=n_previous, n_valid=n,
            unconditional_guidance_scale=args.cfg,
            guidanc_erescale=args.gr,
            ddim_steps=args.ddim_steps, 
            saving_tag="", saving_fps=10
        )
        torch.cuda.empty_cache()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="help document")

    parser.add_argument(
        "--video_path",  type=str, default="/mnt/mnt/public/zefang/AgiBotWorldChallengeIROS2025-WorldModelBaseline/rlinf_dataset_split/val_data/step_0/video/eval/seed_0/rgb.npy",
        help="Path to the input image file"
    )
    parser.add_argument(
        "--input_path",  type=str, default="/mnt/mnt/public/zefang/AgiBotWorldChallengeIROS2025-WorldModelBaseline/rlinf_dataset_split/val_data/step_0/video/eval/seed_0/abs_actions.npy",
        help="Path to the input image file"
    )
    parser.add_argument(
        "--save_root", "-s", type=str, 
        help="Path to save predictions"
    )
    parser.add_argument(
        "--ckp_path", type=str,
    )
    parser.add_argument(
        "--config_path", type=str,
    )

    parser.add_argument(
        "--n_chunk", type=int, default=-1,
        help="number of chunks to predict"
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=27,
    )
    parser.add_argument(
        "--cfg", type=float, default=7.5,
        help="unconditional guidance scale ",
    )
    parser.add_argument(
        "--gr", type=float, default=0.7,
        help="guidance rescale",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda:0"
    )
    parser.add_argument(
        "--seed", type=int,
        default=12345
    )

    args = parser.parse_args()

    main(args)