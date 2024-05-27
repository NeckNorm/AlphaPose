import torch
import cv2
import matplotlib.pyplot as plt

from simple_pose2d_model import get_pose2d_model

from simple_detection_loader import get_detection_model
from simple_detection_loader import DetectionOpt

from simple_pose3d_model import get_pose3d_model
from simple_pose3d_visualizer import get_pose3d_result

from simpe_datawriter import DataWriter

from MotionBERT.lib.data.dataset_wild import WildDetDataset

if __name__ == "__main__":
    DEVICE = "mps"

    pose2d_model_config, pose2d_model = get_pose2d_model(
        pose2d_model_config_path="configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml",
        pose2d_checkpoint_path="pretrained_models/halpe26_fast_res50_256x192.pth",
        device=DEVICE
    )

    detection_model = get_detection_model(
        device=DEVICE, 
        pose2d_model_config=pose2d_model_config
    )

    pose3d_model = get_pose3d_model(
        pose3d_config_path="MotionBERT/configs/pose3d/MB_ft_h36m.yaml",
        pose3d_weight_path="MotionBERT/weights/MB_ft_h36m.bin",
        device=DEVICE
    )

    pose2d_estimator = DataWriter(
        pose2d_model_config, 
        DetectionOpt(DEVICE)
    )

    ########################### Inference ###########################
    img = cv2.imread("examples/demo/custom_example_3.jpeg")

    img_h, img_w = img.shape[:2]

    # Detection
    with torch.no_grad():
        # inps, orig_img, boxes, scores, ids, cropped_boxes
        detection_outp = detection_model.process(img).read()

    (inps, orig_img, boxes, scores, ids, cropped_boxes) = detection_outp

    # 2D Pose Estimation
    with torch.no_grad():
        hm = pose2d_model(inps.to("mps")).cpu()

    pose2d_estimator.save(boxes, scores, ids, hm, cropped_boxes, orig_img)
    hp_outp = pose2d_estimator.start()

    with torch.no_grad():
        hm = pose2d_model(inps.to("mps")).cpu()

    pose2d_estimator.save(boxes, scores, ids, hm, cropped_boxes, orig_img)
    hp_outp = pose2d_estimator.start()

    # 3D Pose Estimation
    keypoints = torch.concat([hp_outp["result"][0]['keypoints'], hp_outp["result"][0]['kp_score']], dim=-1)
    keypoints = torch.unsqueeze(keypoints, dim=0)

    wild_dataset = WildDetDataset(
        clip_len=1,
        image_size=img.shape[:2][::-1],
    )
    wild_dataset.add_data(keypoints)
    test_2d_kp = torch.FloatTensor(wild_dataset[0][0])
    test_2d_kp = torch.unsqueeze(test_2d_kp, dim=0)

    with torch.no_grad():
        pose3d_outp = pose3d_model(test_2d_kp.to(DEVICE)).cpu()

    get_pose3d_result(
        pose3d_outp_single=pose3d_outp[0],
        elevation=12.,
        angle=80
    )
    plt.show()
