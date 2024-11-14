import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

from threading import Thread
from queue import Queue

from simple_pose2d_model import get_pose2d_model
from simple_detection_loader import get_detection_model
from simple_pose3d_model import get_pose3d_model
from custom_model_utils.simple_datawriter import DataWriter
from simple_detection_loader import DetectionOpt
from simple_pose3d_visualizer import get_pose3d_result

from MotionBERT.lib.data.dataset_wild import WildDetDataset
from MotionBERT.lib.utils.vismo import pixel2world_vis_motion

from FrontServer.utils.image_encoder import image_to_bytestream

class Combine:
    def __init__(self):
        DEVICE = "mps"

        pose2d_model_config, self.pose2d_model = get_pose2d_model(
            pose2d_model_config_path="configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml",
            pose2d_checkpoint_path="pretrained_models/halpe26_fast_res50_256x192.pth",
            device=DEVICE
        )

        self.detection_model = get_detection_model(
            device=DEVICE, 
            pose2d_model_config=pose2d_model_config
        )


        self.pose3d_model = get_pose3d_model(
            pose3d_config_path="MotionBERT/configs/pose3d/MB_ft_h36m.yaml",
            pose3d_weight_path="MotionBERT/weights/MB_ft_h36m.bin",
            device=DEVICE
        )

        self.pose2d_estimator = DataWriter(
            pose2d_model_config, 
            DetectionOpt(DEVICE)
        )

        self.vf = self.vid_frame()

        self.original_imges = Queue()
        self.pose3d = Queue()

    def hi(self):
        pose2d_inputs = []
        fig = plt.figure(0, figsize=(10, 10))
        while True:
            img = next(self.vf)
            
            with torch.no_grad():
                detection_outp = self.detection_model.process(img).read()

            (inps, orig_img, boxes, scores, ids, cropped_boxes) = detection_outp
            pose2d_input = (inps[0:1], orig_img, boxes[0:1], scores[0:1], ids[0:1], cropped_boxes[0:1])

            pose2d_results = None
            print(len(pose2d_inputs))
            if len(pose2d_inputs) < 243:
                pose2d_inputs.append(pose2d_input)
                continue
            else:
                pose2d_results = pose2d_inputs
                pose2d_inputs = pose2d_inputs[1:]
                pose2d_inputs.append(pose2d_input)
            
            pose3d_inputs = []
            with torch.no_grad():
                for detection_outp in pose2d_results:
                    (inps, orig_img, boxes, scores, ids, cropped_boxes) = detection_outp
                    hm = self.pose2d_model(inps.to("mps")).cpu()
                    self.pose2d_estimator.save(boxes, scores, ids, hm, cropped_boxes, orig_img)
                    hp_outp = self.pose2d_estimator.start()
                    pose3d_inputs.append(hp_outp)
            
            keypoints = [torch.concat([hp_outp["result"][0]['keypoints'], hp_outp["result"][0]['kp_score']], dim=-1) for hp_outp in pose3d_inputs]
            keypoints = torch.stack(keypoints, dim=0)

            wild_dataset = WildDetDataset(
                clip_len=keypoints.shape[0],
                image_size=pose2d_inputs[0][1].shape[:2][::-1],
            )

            for keypoint in keypoints:
                wild_dataset.add_data(torch.unsqueeze(keypoint, dim=0))

            keypoints2d = np.concatenate(wild_dataset.frames, axis=0)
            keypoints_scores = keypoints2d[...,2]
            keypoints2d = torch.FloatTensor(keypoints2d)
            keypoints2d = torch.unsqueeze(keypoints2d, dim=0)

            with torch.no_grad():
                pose3d_outp = self.pose3d_model(keypoints2d.to("mps")).cpu()[0]

            # p3 = get_pose3d_result(
            #     fig,keypoints_scores[-1], pose3d_outp[-2:-1],80,0
            # )

            p3 = np.zeros((950, 950, 3), np.uint8)

            self.original_imges.put(orig_img)
            self.pose3d.put(p3)

    def start(self):
        self.worker = Thread(
            target=self.hi,
        )
        self.worker.start()

    def get_original_reader(self):
        def _reader():
            while True:
                orig_img = self.original_imges.get()

                yield image_to_bytestream(orig_img, "jpeg")
        
        return _reader()

    def get_pose3d_reader(self):
        def _reader():
            while True:
                pose3d_output = self.pose3d.get()
                yield image_to_bytestream(pose3d_output, "png")
        
        return _reader()

    def vid_frame(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
        while True:
            ret, img = cap.read()

            if not ret:
                break
                
            yield img