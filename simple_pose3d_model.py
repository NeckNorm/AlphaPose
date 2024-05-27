import sys
from os import path

from alphapose.utils.config import update_config

import torch

sys.path.append(path.abspath("./MotionBERT/"))

from collections import OrderedDict
from MotionBERT.lib.utils.learning import load_backbone

class DSTFormerOptions:
    def __init__(self, config_dict):
        self.backbone = "DSTformer"
        self.maxlen = config_dict["maxlen"]
        self.dim_feat = config_dict["dim_feat"]
        self.mlp_ratio = config_dict["mlp_ratio"]
        self.depth = config_dict["depth"]
        self.dim_rep = config_dict["dim_rep"]
        self.num_heads = config_dict["num_heads"]
        self.att_fuse = config_dict["att_fuse"]
        self.num_joints = config_dict["num_joints"]

def get_pose3d_model(pose3d_config_path, pose3d_weight_path, device):
    pose3d_model_config = update_config(pose3d_config_path)

    dstformer_opt = DSTFormerOptions(pose3d_model_config)

    checkpoint_ = torch.load(pose3d_weight_path, map_location=lambda storage, loc: storage)

    checkpoint = OrderedDict()

    for k in checkpoint_['model_pos'].keys():
        new_key = ".".join(k.split(".")[1:])
        checkpoint[new_key] = checkpoint_['model_pos'][k]
    
    pose3d_model = load_backbone(dstformer_opt)
    pose3d_model.load_state_dict(checkpoint, strict=True)
    pose3d_model.to(device)
    pose3d_model.eval()

    print("3D pose model ready!")

    return pose3d_model