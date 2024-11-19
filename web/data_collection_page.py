import justpy as jp

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

import base64
import asyncio
from hashlib import md5
import time
from datetime import datetime
import json

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("../")

from custom_model_utils import DataWriter, get_pose2d_model, get_detection_model, get_pose3d_model, DetectionOpt
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
    # 웹캠 열기
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    cam_fps = cam.get(cv2.CAP_PROP_FPS)
    current_fps = cam_fps

    pose2d_outs = []

    while webpage.is_webcam_on:
        ret, frame = cam.read()

        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break
        
        # =================== Image --> Pose 2D ===================
        pose2d_input = img2pose2d_input(frame)
        pose2d_out = det2pose2d(pose2d_input)

        pose2d_outs.append(pose2d_out)

        pose3d_batch = webpage.collected_data[-1]["batch_size"] if webpage.is_collection_on else 3
        if len(pose2d_outs) > pose3d_batch:
            pose2d_outs = pose2d_outs[1:]
        
        # =================== Pose 2D --> Pose 3D ===================
        img_wh = pose2d_input[1].shape[:2][::-1]
        pose3d_out, keypoints_scores = pose2d_to_pose3d(pose2d_outs, img_wh)

        motion = np.transpose(pose3d_out, (1,2,0))
        motion_world = pixel2world_vis_motion(motion, dim=3)

        # =================== 3D visualize ===================
        f = plt.figure(figsize=(9, 4))
        
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
        img_jpeg = base64.b64encode(jpeg)
        jpg_as_text = img_jpeg.decode('utf-8')

        webcam_img.src = f'data:image/jpeg;base64,{jpg_as_text}' # Data URI 형식으로 웹캠 이미지 설정
        
        # 페이지에 변경 사항 적용
        jp.run_task(webpage.update())

        # =================== Data Collection ===================
        if webpage.is_collection_on:
            collected_data = {
                "index": len(webpage.collected_data[-1]["datas"]),
                "img_jpeg": jpg_as_text,
                "pose3d_output": motion_world.cpu().numpy().tolist(),
                "keypoints_scores": keypoints_scores.tolist()
            }
            webpage.collected_data[-1]["datas"].append(collected_data)
            
            progress_percentage = len(webpage.collected_data[-1]["datas"]) / webpage.collected_data[-1]["frame_count"]
            webpage.update_progress_bar(progress_percentage)

            if progress_percentage == 1:
                webpage.add_collected_item(webpage.collected_data[-1])
                webpage.collecting_off()
        
        # N초 대기
        current_fps = webpage.collected_data[-1]["fps"] if webpage.is_collection_on else current_fps
        await asyncio.sleep(1/current_fps)

    cam.release()
    pose2d_outs = []

    print("웹캠이 닫혔습니다.")

def result_view(container):
    # 웹캠 이미지 표시할 컨테이너 설정
    webcam_container = jp.Div(a=container, id="webcam_container" ,classes="py-10 flex flex-col border-box justify-center items-center bg-white")

    # 이미지 표시 영역 생성
    jp.Img(a=webcam_container, id="webcam_img", classes="block w-full max-w-3xl", width="640", height="480")

    # 3D 결과 ploting
    jp.Matplotlib(a=webcam_container, id="pose3d_figures", classes="w-full max-h-80")

def progress_bar_view(container):
    progress_bar_container = jp.Div(a=container, id="progress_bar_container", classes="relative w-full h-6 flex justify-center items-center bg-white border-2 border-black mt-10 z-0")
    progress_bar = jp.Div(a=progress_bar_container, id="progress_bar", classes="absolute left-0 h-5 bg-yellow-300 z-0")
    progress_text = jp.Span(a=progress_bar_container, text="0%", id="progress_text", classes="z-20")

    def update(percentage):
        percentage = percentage * 100
        progress_bar.style = f"width: {percentage}%"
        progress_text.text = str(percentage) + "%"
    
    container.a.a.update_progress_bar = update

def setting_view(container):
    # =================== 설정 ===================
    setting_container = jp.Div(a=container, id="setting_container", classes="h-full flex flex-col border-box mx-5")

    setting_controller = jp.Div(a=setting_container, id="setting_controller", classes="flex flex-col items-end bg-gray-300 p-5 rounded-lg mt-10")
    jp.Span(a=setting_controller, text="데이터 생성 설정", classes="text-lg self-start font-bold mb-3")


    # =================== 웹캠 정보 가져오기 ===================
    cam = cv2.VideoCapture(0)
    setting_container.width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    setting_container.height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    setting_container.cam_fps = cam.get(cv2.CAP_PROP_FPS)
    setting_container.user_fps = setting_container.cam_fps
    setting_container.batch = 3
    cam.release()

    # =================== 생성 시간 ===================
    def time_control(self, msg):
        time_value = float(self.value)
        user_fps = self.a.a.a.user_fps
        self.a.a.components[2].components[-1].value = time_value * user_fps  if user_fps > 0 else 0

        self.a.a.a.time_length = time_value
        self.a.a.a.frame_count = self.a.a.components[2].components[-1].value
    
    setting_control_time = jp.Div(a=setting_controller, id="setting_control_time", classes="flex justify-around items-center")
    jp.Span(a=setting_control_time, text="생성 길이(초)", classes="text-base")
    jp.Input(a=setting_control_time, id="setting_control_time_input", placeholder="초 단위로 입력", classes="m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500")
    setting_control_time.components[-1].on("input", time_control)

    # =================== 생성 프레임 ===================
    def frame_control(self, msg):
        frame_value = float(self.value)
        user_fps = self.a.a.a.user_fps
        self.a.a.components[1].components[-1].value = frame_value / user_fps if user_fps > 0 else 0

        self.a.a.a.frame_count = frame_value
        self.a.a.a.time_length = self.a.a.components[1].components[-1].value

    setting_control_frame = jp.Div(a=setting_controller, id="setting_control_frame", classes="flex justify-around items-center")
    jp.Span(a=setting_control_frame, text="생성 프레임 개수", classes="text-base")
    jp.Input(a=setting_control_frame, id="setting_control_frame_input", placeholder="프레임 개수를 입력", classes="m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500")
    setting_control_frame.components[-1].on("input", frame_control)
    
    # =================== 3D pose 배치 크기 ===================
    def batch_control(self, msg):
        batch_value = int(self.value)
        batch_value = batch_value if batch_value > 0 else 1

        self.a.a.a.batch = batch_value

    setting_control_batch = jp.Div(a=setting_controller, id="setting_control_batch", classes="flex justify-around items-center")
    jp.Span(a=setting_control_batch, text="3D pose 배치 크기", classes="text-base")
    jp.Input(a=setting_control_batch, id="setting_control_batch_input", placeholder="배치 크기 입력", classes="m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500")
    setting_control_batch.components[-1].on("input", batch_control)

    # =================== 화면 캡처 주기 ===================
    def capture_interval_control(self, msg):
        user_fps = float(self.value)
        user_fps = user_fps if user_fps > 0 else 0
        if user_fps == 0:
            return

        # 생성 시간
        user_frame = self.a.a.components[2].components[-1].value
        self.a.a.components[1].components[-1].value = user_frame / user_fps

        # 값 업데이트
        self.a.a.a.time_length = self.a.a.components[1].components[-1].value
        self.a.a.a.user_fps = user_fps

    setting_control_capture_interval = jp.Div(a=setting_controller, id="setting_control_capture_interval", classes="flex justify-around items-center")
    jp.Span(a=setting_control_capture_interval, text="화면 캡처 주기(fps)", classes="text-base")
    jp.Input(a=setting_control_capture_interval, id="setting_control_capture_interval_input", placeholder="fps 입력", classes="m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500")
    setting_control_capture_interval.components[-1].on("input", capture_interval_control)
    setting_control_capture_interval.components[-1].value = setting_container.cam_fps
    setting_control_capture_interval.components[-2].value = 3

    # =================== 웹캠 정보 ===================
    current_webcam_info_container = jp.Div(a=setting_controller, id="current_webcam_info_container", classes="w-full flex justify-around items-center my-2")
    jp.Span(a=current_webcam_info_container, text=f"웹캠 fps : {setting_container.cam_fps}", classes="text-sm text-purple-500 font-bold")
    jp.Span(a=current_webcam_info_container, text=f"이미지 크기 : {setting_container.width} x {setting_container.height}", classes="text-sm text-purple-500 font-bold")

    # =================== Progress Bar ===================
    progress_bar_view(setting_container)

    # =================== 웹캠 및 수집 버튼 ===================
    def webcam_control(self, msg):
        def webcam_off():
            container.a.is_webcam_on = False
            self.set_class('bg-red-600')
            self.text = "웹캠 키기"

        def webcam_on():
            container.a.is_webcam_on = True
            self.text = "웹캠 끄기"
            self.set_class('bg-purple-400')
            
            webpage = container.a
            webcam_img = container.components[0].components[0]
            pose3d_figures = container.components[0].components[1]
            jp.run_task(update_webcam(webpage, webcam_img, pose3d_figures))

        if msg is None:
            container.a.webcam_off = webcam_off
            container.a.webcam_on = webcam_on
        else:
            if container.a.is_webcam_on:
                webcam_off()
            else:
                webcam_on()

    def collection_control(self, msg):
        def collecting_off():
            container.a.update_progress_bar(0)

            container.a.is_collection_on = False
            self.set_class('bg-yellow-600')
            self.text = "데이터 수집하기"
        
        def collecting_on():
            if not(container.a.is_webcam_on):
                container.a.webcam_on()

            container.a.update_progress_bar(0)

            self.text = "수집 종료하기"
            self.set_class('bg-pink-600')
            
            collection_box = {
                "date": datetime.now().strftime("%Y-%m-%d(%H:%M:%S)"),
                "hash": md5(str(time.time()).encode("utf-8")).hexdigest(),
                "img_width": self.a.a.width,
                "img_height": self.a.a.height,
                "frame_count": self.a.a.frame_count,
                "batch_size": self.a.a.batch,
                "fps": self.a.a.user_fps,
                "time_length": self.a.a.time_length,
                "datas": []
            }

            container.a.collected_data.append(collection_box)

            container.a.is_collection_on = True

        if msg is None:
            container.a.collecting_off = collecting_off
            container.a.collecting_on = collecting_on
        else:
            if container.a.is_collection_on:
                collecting_off()
            else:
                collecting_on()

    core_button_container = jp.Div(a=setting_container, id="core_button_container", classes="flex justify-center mt-5")
    container.a.is_webcam_on = False
    container.a.is_collection_on = False

    jp.Button(a=core_button_container, id="webcam_control_btn", text="웹캠 키기", classes="w-32 m-2 bg-red-600 hover:bg-red-100 hover:text-black text-white font-bold py-3 px-4 rounded-lg")
    jp.Button(a=core_button_container, id="data_collection_control_btn", text="데이터 수집하기", classes="w-35 m-2 bg-yellow-600 hover:bg-red-100 hover:text-black text-white font-bold py-3 px-4 rounded-lg")

    # 초기화
    webcam_control(core_button_container.components[0], None)
    collection_control(core_button_container.components[1], None)

    core_button_container.components[0].on("click", webcam_control)
    core_button_container.components[1].on("click", collection_control)

def download_item(container, collected_datas):
    """
    collection_box = {
        "date": datetime.now().strftime("%Y-%m-%d(%H:%M)"),
        "hash": md5(str(time.time()).encode("utf-8")).hexdigest(),
        "img_width": self.a.a.width,
        "img_height": self.a.a.height,
        "frame_count": self.a.a.frame_count,
        "batch_size": self.a.a.batch,
        "fps": self.a.a.user_fps,
        "time_length": self.a.a.time_length,
        "datas": []
    }

    collected_data = {
        "index": len(webpage.collected_data[-1]["datas"]),
        "img_jpeg": img_jpeg,
        "pose3d_output": motion_world,
        "keypoints_scores": keypoints_scores
    }
    """
    item_container = jp.Div(a=container, classes="h-15 flex p-3 items-center bg-green-100 mb-3")
    date = collected_datas["date"]
    frame_count = int(collected_datas["frame_count"])
    batch_size = collected_datas["batch_size"]
    item_name = f"{date}_fc{frame_count}_bs{batch_size}"
    jp.Span(a=item_container, text=item_name, classes="text-base mr-5")
    download_btn = jp.A(a=item_container, text="💾", classes="text-2xl cursor-pointer")

    hash = collected_datas["hash"]
    
    selected_data = None
    for data in container.a.a.a.collected_data:
        if data["hash"] == hash:
            selected_data = data
            break
    
    json_file = item_name + ".json"
    with open(json_file, "w") as f:
        json.dump(selected_data, f)
    
    download_btn.href=f"/static/{json_file}"
    download_btn.download=json_file

def download_view(container):
    download_container = jp.Div(a=container, id="download_container", classes="h-full flex flex-col items-center border-box bg-white border border-green-400 p-5")
    jp.Span(a=download_container, text="저장된 데이터 목록", classes="text-lg font-bold mb-1")

    download_list_container = jp.Div(a=download_container, id="download_list_container", classes="h-full flex flex-col")

    def add_collected_item(collected_datas):
        download_item(download_list_container, collected_datas)
    
    container.a.add_collected_item = add_collected_item

def data_collection_page():
    wp = jp.WebPage()
    wp.collected_data = []

    plane = jp.Div(a=wp, id="plane", classes="h-screen w-screen flex border-box justify-center py-3")
    
    result_view(plane)
    setting_view(plane)
    download_view(plane)

    return wp

async def update_result(webpage, webcam_img, pose3d_figures):
    # 웹캠 열기
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    cam_fps = cam.get(cv2.CAP_PROP_FPS)
    current_fps = cam_fps

    pose2d_outs = []

    while webpage.is_webcam_on:
        ret, frame = cam.read()

        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break
        
        # =================== Image --> Pose 2D ===================
        pose2d_input = img2pose2d_input(frame)
        pose2d_out = det2pose2d(pose2d_input)

        pose2d_outs.append(pose2d_out)

        pose3d_batch = webpage.collected_data[-1]["batch_size"] if webpage.is_collection_on else 3
        if len(pose2d_outs) > pose3d_batch:
            pose2d_outs = pose2d_outs[1:]
        
        # =================== Pose 2D --> Pose 3D ===================
        img_wh = pose2d_input[1].shape[:2][::-1]
        pose3d_out, keypoints_scores = pose2d_to_pose3d(pose2d_outs, img_wh)

        motion = np.transpose(pose3d_out, (1,2,0))
        motion_world = pixel2world_vis_motion(motion, dim=3)

        # =================== 3D visualize ===================
        f = plt.figure(figsize=(9, 4))
        
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
        img_jpeg = base64.b64encode(jpeg)
        jpg_as_text = img_jpeg.decode('utf-8')

        webcam_img.src = f'data:image/jpeg;base64,{jpg_as_text}' # Data URI 형식으로 웹캠 이미지 설정
        
        # 페이지에 변경 사항 적용
        jp.run_task(webpage.update())

        # =================== Data Collection ===================
        if webpage.is_collection_on:
            collected_data = {
                "index": len(webpage.collected_data[-1]["datas"]),
                "img_jpeg": jpg_as_text,
                "pose3d_output": motion_world.cpu().numpy().tolist(),
                "keypoints_scores": keypoints_scores.tolist()
            }
            webpage.collected_data[-1]["datas"].append(collected_data)
            
            progress_percentage = len(webpage.collected_data[-1]["datas"]) / webpage.collected_data[-1]["frame_count"]
            webpage.update_progress_bar(progress_percentage)

            if progress_percentage == 1:
                webpage.add_collected_item(webpage.collected_data[-1])
                webpage.collecting_off()
        
        # N초 대기
        current_fps = webpage.collected_data[-1]["fps"] if webpage.is_collection_on else current_fps
        await asyncio.sleep(1/current_fps)

    cam.release()
    pose2d_outs = []

    print("웹캠이 닫혔습니다.")

if __name__ == "__main__":
    jp.justpy()