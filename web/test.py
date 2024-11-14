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
# pose2d_model, detection_model, pose3d_model, pose2d_estimator = load_models()
run_webcam = False

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

async def update_webcam(webpage, webcam_img, pose3d_figures):
    global run_webcam
    # 웹캠 열기
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    pose2d_outs = []

    while run_webcam:
        ret, frame = cam.read()

        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break
        
        # =================== Image --> Pose 2D ===================
        pose2d_input = img2pose2d_input(frame)
        pose2d_out = det2pose2d(pose2d_input)

        pose2d_outs.append(pose2d_out)

        if len(pose2d_outs) > 3:
            pose2d_outs = pose2d_outs[1:]
        
        # =================== Pose 2D --> Pose 3D ===================
        img_wh = pose2d_input[1].shape[:2][::-1]
        pose3d_out, keypoints_scores = pose2d_to_pose3d(pose2d_outs, img_wh)

        motion = np.transpose(pose3d_out, (1,2,0))
        motion_world = pixel2world_vis_motion(motion, dim=3)

        # =================== 3D visualize ===================
        f = plt.figure(figsize=(10, 5))
        
        ax = f.add_subplot(131, projection='3d')
        pose3d_visualize(ax, motion_world, keypoints_scores, 80, 0)
        plt.title("TOP VIEW")

        ax = f.add_subplot(132, projection='3d')
        pose3d_visualize(ax, motion_world, keypoints_scores, 40, -90)
        plt.title("FRONT VIEW")

        ax = f.add_subplot(133, projection='3d')
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
        jp.run_task(webpage.update())
        
        # 0.1초 대기 후 다음 프레임
        await asyncio.sleep(0.01)

    cam.release()
    pose2d_outs = []

    print("웹캠이 닫혔습니다.")

def toggle_webcam(self, msg):
    global run_webcam
    if run_webcam:
        run_webcam = False
        self.text = "시작하기"
    else:
        run_webcam = True
        jp.run_task(update_webcam(self.a.a, self.webcam_img, self.pose3d_figures))
        self.text = "멈추기"

def result_view(container):
    # 웹캠 이미지 표시할 컨테이너 설정
    webcam_container = jp.Div(a=container, id="webcam_container" ,classes="py-10 flex flex-col border-box justify-center items-center bg-white")

    # 이미지 표시 영역 생성
    jp.Img(a=webcam_container, id="webcam_img", classes="border", width="640", height="480")

    # 3D 결과 ploting
    jp.Matplotlib(a=webcam_container, id="pose3d_figures")

def setting_view(container):
    # =================== 웹캠 정보 가져오기 ===================
    cam = cv2.VideoCapture(0)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cam.get(cv2.CAP_PROP_FPS)
    cam.release()

    # =================== 설정 ===================
    setting_container = jp.Div(a=container, id="setting_container", classes="h-full flex flex-col border-box bg-red-50")

    setting_controller = jp.Div(a=setting_container, id="setting_controller", classes="flex flex-col items-end bg-gray-300 p-5 rounded-lg")
    jp.Span(a=setting_controller, text="데이터 생성 설정", classes="text-lg self-start font-bold mb-3")

    # =================== 생성 시간 ===================
    def time_control(self, msg):
        time_value = float(self.value)
        self.a.a.components[2].components[-1].value = time_value * fps  if fps > 0 else "fps가 0입니다"
    
    setting_control_time = jp.Div(a=setting_controller, id="setting_control_time", classes="flex justify-around items-center")
    jp.Span(a=setting_control_time, text="생성 길이(초)", classes="text-base")
    jp.Input(a=setting_control_time, id="setting_control_time_input", placeholder="초 단위로 입력", classes="m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500")
    setting_control_time.components[-1].on("input", time_control)

    # =================== 생성 프레임 ===================
    def frame_control(self, msg):
        frame_value = float(self.value)
        self.a.a.components[1].components[-1].value = frame_value / fps if fps > 0 else "fps가 0입니다"

    setting_control_frame = jp.Div(a=setting_controller, id="setting_control_frame", classes="flex justify-around items-center")
    jp.Span(a=setting_control_frame, text="생성 프레임 개수", classes="text-base")
    jp.Input(a=setting_control_frame, id="setting_control_frame_input", placeholder="프레임 개수를 입력", classes="m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500")
    setting_control_frame.components[-1].on("input", frame_control)
    
    # =================== 3D pose 배치 크기 ===================
    setting_control_batch = jp.Div(a=setting_controller, id="setting_control_batch", classes="flex justify-around items-center")
    jp.Span(a=setting_control_batch, text="3D pose 배치 크기", classes="text-base")
    jp.Input(a=setting_control_batch, id="setting_control_batch_input", placeholder="배치 크기 입력", classes="m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500")
    
    # =================== 웹캠 정보 ===================
    current_webcam_info_container = jp.Div(a=setting_controller, id="current_webcam_info_container", classes="w-full flex justify-around items-center my-2")
    jp.Span(a=current_webcam_info_container, text=f"웹캠 fps : {fps}", classes="text-sm text-purple-500 font-bold")
    jp.Span(a=current_webcam_info_container, text=f"이미지 크기 : {width} x {height}", classes="text-sm text-purple-500 font-bold")

    # =================== 저장 버튼 ===================
    save_button_container = jp.Div(a=setting_controller, id="save_button_container", classes="flex justify-center w-full")
    jp.Button(a=save_button_container, id="setting_save_btn", text="저장하기", classes="w-32 m-2 bg-green-400 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg")

def main():
    wp = jp.WebPage()

    plane = jp.Div(a=wp, id="plane", classes="h-screen w-screen flex border-box justify-center bg-indigo-500")
    
    result_view(plane)
    setting_view(plane)

    # toggle_btn = jp.Button(a=webcam_container, click=toggle_webcam, text="시작하기")
    # toggle_btn.pose3d_figures = pose3d_figures
    # toggle_btn.webcam_img = webcam_img

    return wp

if __name__ == "__main__":
    jp.justpy(main)