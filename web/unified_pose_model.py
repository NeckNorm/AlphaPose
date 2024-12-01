import torch
import numpy as np

from custom_model_utils import DataWriter, get_pose2d_model, get_detection_model, get_pose3d_model, DetectionOpt, get_pose2d_result
from MotionBERT.lib.data.dataset_wild import WildDetDataset

CONFIGS = {
    "pose2d": {
        "config": "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml",
        "checkpoint": "pretrained_models/halpe26_fast_res50_256x192.pth"
    },
    "pose3d": {
        "config": "../MotionBERT/configs/pose3d/MB_ft_h36m.yaml",
        "checkpoint": "../MotionBERT/weights/MB_ft_h36m.bin"
    }
}

class UnifiedPoseModel:
    def __init__(self, device:str="mps"):
        
        pose2d_model_config, pose2d_model = get_pose2d_model(
            pose2d_model_config_path=CONFIGS["pose2d"]["config"],
            pose2d_checkpoint_path=CONFIGS["pose2d"]["checkpoint"],
            device=device,
            root_path="../"
        )

        detection_model = get_detection_model(
            device=device, 
            pose2d_model_config=pose2d_model_config
        )

        pose3d_model = get_pose3d_model(
            pose3d_config_path=CONFIGS["pose3d"]["config"],
            pose3d_weight_path=CONFIGS["pose3d"]["checkpoint"],
            device=device
        )

        pose2d_estimator = DataWriter(
            pose2d_model_config, 
            DetectionOpt(device)
        )

        self.device = device
        self.pose2d_model = pose2d_model
        self.detection_model = detection_model
        self.pose3d_model = pose3d_model
        self.pose2d_estimator = pose2d_estimator

        self.clip = list()
        
    def img_to_detection(self, img):
        detection_outp = self.detection_model.process(img).read()

        return detection_outp

    def detection_to_pose2d(self, detection_outp, draw_2d_output:bool=False):
        (inps, orig_img, boxes, scores, ids, cropped_boxes) = detection_outp

        # We assume that the input image contains only one person
        # So if there are multiple persons, output performance will be degraded
        inps            = inps[0:1]
        boxes           = boxes[0:1]
        scores          = scores[0:1]
        ids             = ids[0:1]
        cropped_boxes   = cropped_boxes[0:1]

        with torch.no_grad():
            hm = self.pose2d_model(inps.to(self.device)).cpu()
            self.pose2d_estimator.save(boxes, scores, ids, hm, cropped_boxes, orig_img)

            hp_outp = self.pose2d_estimator.start()

        if draw_2d_output:
            return hp_outp, get_pose2d_result(orig_img, hp_outp)
        else:
            return hp_outp, orig_img
    
    def pose2d_to_pose3d(self, pose2d_outs, img_wh):
        keypoints = [torch.concat([pose2d_out["result"][0]['keypoints'], pose2d_out["result"][0]['kp_score']], dim=-1) for pose2d_out in pose2d_outs]
        keypoints = torch.stack(keypoints, dim=0)

        wild_dataset = WildDetDataset(
            clip_len=keypoints.shape[0],
            image_size=img_wh,
        )
        for keypoint in keypoints:
            wild_dataset.add_data(keypoint[None,...])
        
        keypoints_transformed = np.concatenate(wild_dataset.frames, axis=0)

        keypoints_scores = keypoints_transformed[...,2] # (T, 17)

        keypoints_transformed = torch.FloatTensor(keypoints_transformed)
        keypoints_transformed = keypoints_transformed[None,...] # (1, T, 17, 3)
        keypoints_2d = keypoints_transformed[0,...,:2] # (T, 17, 2)

        with torch.no_grad():
            pose3d_outp = self.pose3d_model(keypoints_transformed.to(self.device)).cpu()[0] # We use only batch size 1, shape = (T, 17, 3)

        return pose3d_outp, keypoints_scores, keypoints_2d
    
    def clear_clip(self):
        self.clip = list()
    
    def add_clip(self, clip):
        self.clip.append(clip)

    def remove_first_clip(self):
        self.clip = self.clip[1:]

    def current_clip_length(self):
        return len(self.clip)

    def __call__(self, img, target_clip_length:int=3, draw_2d_output:bool=False):
        detection_outp = self.img_to_detection(img)
        pose2d_outp, input_image = self.detection_to_pose2d(detection_outp, draw_2d_output)

        self.add_clip(pose2d_outp)

        if self.current_clip_length() < target_clip_length:
            return None, None, None, input_image
        
        pose3d_outp, keypoints_scores, keypoints_2d = self.pose2d_to_pose3d(self.clip, img.shape[:2][::-1])

        self.remove_first_clip()
            
        return pose3d_outp, keypoints_scores, keypoints_2d, input_image