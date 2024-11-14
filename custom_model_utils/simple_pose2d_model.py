from alphapose.utils.config import update_config
from alphapose.models import builder
from .simple_detection_loader import get_detection_model

import torch

import threading

def get_pose2d_model(
    pose2d_model_config_path, 
    pose2d_checkpoint_path,
    device,
    root_path = "./"
):
    pose2d_model_config_path = root_path + pose2d_model_config_path
    pose2d_checkpoint_path = root_path + pose2d_checkpoint_path

    pose2d_model_config = update_config(
        config_file = pose2d_model_config_path
    )

    pose2d_model_config.DETECTOR.CONFIG = root_path + pose2d_model_config.DETECTOR.CONFIG
    pose2d_model_config.DETECTOR.WEIGHTS = root_path + pose2d_model_config.DETECTOR.WEIGHTS


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

class Pose2DForServer:
    def __init__(self, pose2d_model_config_path, pose2d_checkpoint_path, device):
        pose2d_config, self.pose2d_model = get_pose2d_model(pose2d_model_config_path, pose2d_checkpoint_path, device)
        self.det_model = get_detection_model(pose2d_model_config=pose2d_config, device=device)

        self.inputs = []
        self.lock = threading.Lock()
    
    def add(self, image):
        self.lock.acquire()
        self.inputs.append(image)
        self.lock.release()
    
    def realtime_reader(self):

        def _realtime_reader():
            while True:
                try:
                    self.lock.acquire()
                    inp = self.inputs.pop()
                    self.lock.release()

                    
                except IndexError as e:
                    continue
                
