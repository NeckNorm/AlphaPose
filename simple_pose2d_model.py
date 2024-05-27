from alphapose.utils.config import update_config
from alphapose.models import builder

import torch

def get_pose2d_model(
    pose2d_model_config_path, 
    pose2d_checkpoint_path,
    device
):
    pose2d_model_config = update_config(
        config_file = pose2d_model_config_path
    )

    pose2d_model = builder.build_sppe(
        cfg = pose2d_model_config.MODEL, 
        preset_cfg=pose2d_model_config.DATA_PRESET
    )
    pose2d_model.load_state_dict(
        torch.load(pose2d_checkpoint_path, map_location=device)
    )

    pose2d_model.to(device=device)
    pose2d_model.eval()
    print("2D Pose model ready!")

    return pose2d_model_config, pose2d_model