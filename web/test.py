import justpy as jp

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

import base64
import asyncio

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("../")

from custom_model_utils import DataWriter, get_pose2d_model, get_detection_model, get_pose3d_model, DetectionOpt
from custom_model_utils import get_pose2d_result, get_pose3d_result
from MotionBERT.lib.data.dataset_wild import WildDetDataset
from MotionBERT.lib.utils.vismo import pixel2world_vis_motion

# CONSTANTS
DEVICE = "mps"

def load_models():
    pose2d_model_config, pose2d_model = get_pose2d_model(
        pose2d_model_config_path="configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml",
        pose2d_checkpoint_path="pretrained_models/halpe26_fast_res50_256x192.pth",
        device=DEVICE,
        root_path="../"
    )

    detection_model = get_detection_model(
        device=DEVICE, 
        pose2d_model_config=pose2d_model_config
    )

    pose3d_model = get_pose3d_model(
        pose3d_config_path="../MotionBERT/configs/pose3d/MB_ft_h36m.yaml",
        pose3d_weight_path="../MotionBERT/weights/MB_ft_h36m.bin",
        device=DEVICE
    )

    pose2d_estimator = DataWriter(
        pose2d_model_config, 
        DetectionOpt(DEVICE)
    )

    return pose2d_model, detection_model, pose3d_model, pose2d_estimator

def pose3d_visualize(ax, motion, scores,elivation, angle, keypoints_threshold=0.7):
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"

    j3d = motion[:,:,0]
    ax.set_xlim(-512, 0)
    ax.set_ylim(-256, 256)
    ax.set_zlim(-512, 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elivation, azim=angle)
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    for i in range(len(joint_pairs)):
        if scores[0][i] < keypoints_threshold:
            continue
        limb = joint_pairs[i]
        xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
        if joint_pairs[i] in joint_pairs_left:
            ax.plot(-xs, -zs, -ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
        elif joint_pairs[i] in joint_pairs_right:
            ax.plot(-xs, -zs, -ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
        else:
            ax.plot(-xs, -zs, -ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization

# GLBAL VARIABLES
pose2d_model, detection_model, pose3d_model, pose2d_estimator = load_models()

wp = jp.WebPage(delete_flag=False)

def img2pose2d_input(img):
    img_h, img_w = img.shape[:2]
    # inps, orig_img, boxes, scores, ids, cropped_boxes
    detection_outp = detection_model.process(img).read()

    (inps, orig_img, boxes, scores, ids, cropped_boxes) = detection_outp
    pose2d_input = (inps[0:1], orig_img, boxes[0:1], scores[0:1], ids[0:1], cropped_boxes[0:1])

    # Bounding box를 보고싶을 경우 주석 해제
    # l,t,r,b = np.array(boxes[0], np.int32)
    # cv2.rectangle(img, (l,t), (r,b), (0,0,255), 3)

    return pose2d_input

def det2pose2d(pose2d_input):
    with torch.no_grad():
        (inps, orig_img, boxes, scores, ids, cropped_boxes) = pose2d_input
        hm = pose2d_model(inps.to("mps")).cpu()
        pose2d_estimator.save(boxes, scores, ids, hm, cropped_boxes, orig_img)
        hp_outp = pose2d_estimator.start()

        # 2D pose 결과 이미지를 보고싶다면 주석을 해제
        # 이후 리턴 값을 랜더링
        # return get_pose2d_result(orig_img, hp_outp)

        return hp_outp

def pose2d_to_pose3d(pose2d_outs, img_wh):
        keypoints = [torch.concat([pose2d_out["result"][0]['keypoints'], pose2d_out["result"][0]['kp_score']], dim=-1) for pose2d_out in pose2d_outs]
        keypoints = torch.stack(keypoints, dim=0)

        wild_dataset = WildDetDataset(
            clip_len=keypoints.shape[0],
            image_size=img_wh,
        )
        for keypoint in keypoints:
            wild_dataset.add_data(keypoint[None,...])
        
        keypoints_transformed = np.concatenate(wild_dataset.frames, axis=0)

        keypoints_scores = keypoints_transformed[...,2]

        keypoints_transformed = torch.FloatTensor(keypoints_transformed)
        keypoints_transformed = keypoints_transformed[None,...]

        with torch.no_grad():
            pose3d_outp = pose3d_model(keypoints_transformed.to(DEVICE)).cpu()[0]

        return pose3d_outp, keypoints_scores

async def update_webcam(webcam_img, pose3d_figures):
    # 웹캠 열기
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    pose2d_outs = []

    while True:
        ret, frame = cam.read()

        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break
        
        # =================== Image --> Pose 2D ===================
        pose2d_input = img2pose2d_input(frame)
        pose2d_out = det2pose2d(pose2d_input)

        pose2d_outs.append(pose2d_out)

        if len(pose2d_outs) > 10:
            pose2d_outs = pose2d_outs[1:]
        
        # =================== Pose 2D --> Pose 3D ===================
        img_wh = pose2d_input[1].shape[:2][::-1]
        pose3d_out, keypoints_scores = pose2d_to_pose3d(pose2d_outs, img_wh)

        motion = np.transpose(pose3d_out, (1,2,0))
        motion_world = pixel2world_vis_motion(motion, dim=3)

        # =================== 3D visualize ===================
        f = plt.figure(figsize=(10, 5))
        
        ax = f.add_subplot(121, projection='3d')
        pose3d_visualize(ax, motion_world, keypoints_scores, 80, 0)
        plt.title("TOP VIEW")

        ax = f.add_subplot(122, projection='3d')
        pose3d_visualize(ax, motion_world, keypoints_scores, 0, 0)
        plt.title("LEFT SIDE VIEW")

        pose3d_figures.set_figure(f)
        plt.close(f)

        pose3d_figures.update()


        # =================== Screen Update ===================
        _, jpeg = cv2.imencode('.jpg', frame) # 이미지를 JPEG로 인코딩 후 base64로 변환
        jpg_as_text = base64.b64encode(jpeg).decode('utf-8')

        webcam_img.src = f'data:image/jpeg;base64,{jpg_as_text}' # Data URI 형식으로 웹캠 이미지 설정
        
        # 페이지에 변경 사항 적용
        jp.run_task(wp.update())
        
        # 0.1초 대기 후 다음 프레임
        await asyncio.sleep(0.01)

    cam.release()
    print("웹캠이 닫혔습니다.")

async def main():
    # 웹캠 이미지 표시할 컨테이너 설정
    webcam_container = jp.Div(a=wp, classes="h-screen w-screen flex flex-col justify-center items-center bg-white")

    pose3d_figures = jp.Matplotlib(a=webcam_container)

    # 이미지 표시 영역 생성
    webcam_img = jp.Img(a=webcam_container, classes="border", width="640", height="480")

    # 페이지 로드 시 웹캠 업데이트 시작
    jp.run_task(update_webcam(webcam_img, pose3d_figures))

    return wp

if __name__ == "__main__":
    jp.justpy(main)