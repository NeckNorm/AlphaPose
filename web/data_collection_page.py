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

from MotionBERT.lib.data.dataset_wild import WildDetDataset
from MotionBERT.lib.utils.vismo import pixel2world_vis_motion
from utils import pose3d_visualize, screen_update
from unified_pose_model import UnifiedPoseModel
from keypoint_adjust_utils import adjust_head_pose, adjust_neck_pose

# CONSTANTS
DEVICE = "mps"
unified_pose_model = UnifiedPoseModel(device=DEVICE)

async def update_webcam(node_dict: dict):
    # 웹캠 열기
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    cam_fps = cam.get(cv2.CAP_PROP_FPS)
    current_fps = cam_fps

    while node_dict["webpage"].is_webcam_on:
        ret, frame = cam.read()

        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break
        
        pose3d_clip_length = node_dict["webpage"].collected_data[-1]["clip_length"] if node_dict["webpage"].is_collection_on else 3

        pose3d_out, keypoints_scores, keypoints_2d, input_image = unified_pose_model(frame, target_clip_length=pose3d_clip_length, draw_2d_output=True)

        # =================== Screen Update ===================
        img_jpeg = screen_update(screen=node_dict["webcam_img"], image=input_image)
        
        if pose3d_out is None:
            jp.run_task(node_dict["webpage"].update())
            continue

        motion = np.transpose(pose3d_out, (1,2,0)) # (17, 3, T)
        motion_world = pixel2world_vis_motion(motion, dim=3)

        motion_world = adjust_head_pose(motion_world, keypoints_2d)
        motion_world = adjust_neck_pose(motion_world)

        # EMA in clip
        motion_world = motion_world.cpu().numpy()
        for idx in range(1, motion_world.shape[2]):
            motion_world[...,idx] = 0.1 * motion_world[...,idx-1] + 0.9 * motion_world[...,idx]

        motion_world = torch.FloatTensor(motion_world)

        # =================== 3D visualize ===================
        f = plt.figure(figsize=(9, 4))
        
        ax = f.add_subplot(131, projection='3d')
        pose3d_visualize(ax, motion_world[...,-1], keypoints_scores[-1], 90, 0)
        plt.title("TOP VIEW")

        ax = f.add_subplot(132, projection='3d')
        pose3d_visualize(ax, motion_world[..., -1], keypoints_scores[-1], 0, -90)
        plt.title("FRONT VIEW")

        ax = f.add_subplot(133, projection='3d')
        pose3d_visualize(ax, motion_world[..., -1], keypoints_scores[-1], 0, 0)
        plt.title("LEFT SIDE VIEW")

        node_dict["pose3d_figures"].set_figure(f)
        plt.close(f)

        node_dict["pose3d_figures"].update()
 
        # 페이지에 변경 사항 적용
        jp.run_task(node_dict["webpage"].update())

        # =================== Data Collection ===================
        if node_dict["webpage"].is_collection_on:
            collected_data = {
                "index"             : len(node_dict["webpage"].collected_data[-1]["datas"]),
                "img_jpeg"          : img_jpeg,
                "pose3d_output"     : motion_world.cpu().numpy().tolist(),
                "keypoints_scores"  : keypoints_scores.tolist()
            }
            node_dict["webpage"].collected_data[-1]["datas"].append(collected_data)
            
            progress_percentage = len(node_dict["webpage"].collected_data[-1]["datas"]) / node_dict["webpage"].collected_data[-1]["frame_count"]
            node_dict["webpage"].update_progress_bar(progress_percentage)

            if progress_percentage == 1:
                node_dict["webpage"].add_collected_item(node_dict["webpage"].collected_data[-1])
                node_dict["webpage"].collecting_off()
                unified_pose_model.clear_clip()
        
        # N초 대기
        current_fps = node_dict["webpage"].collected_data[-1]["fps"] if node_dict["webpage"].is_collection_on else current_fps
        await asyncio.sleep(1/current_fps)

    cam.release()
    pose2d_outs = []

    print("웹캠이 닫혔습니다.")

def result_view(node_dict: dict):
    # 웹캠 이미지 표시할 컨테이너 설정
    webcam_container = jp.Div(
        a           = node_dict["plane"], 
        id          = "webcam_container",
        classes     = "py-10 flex flex-col border-box justify-center items-center bg-white"
    )
    node_dict["webcam_container"] = webcam_container

    # 이미지 표시 영역 생성
    webcam_img = jp.Img(
        a           = webcam_container, 
        id          = "webcam_img", 
        classes     = "block w-full max-w-3xl",
        width       = "640",
        height      = "480"
    )
    node_dict["webcam_img"] = webcam_img
    
    # 3D 결과 ploting
    pose3d_figures = jp.Matplotlib(
        a           = webcam_container, 
        id          = "pose3d_figures", 
        classes     = "w-full max-h-80"
    )
    node_dict["pose3d_figures"] = pose3d_figures

def progress_bar_view(container, node_dict: dict):
    progress_bar_container = jp.Div(
        a           = container, 
        id          = "progress_bar_container", 
        classes     = "relative w-full h-6 flex justify-center items-center bg-white border-2 border-black mt-10 z-0"
    )
    progress_bar = jp.Div(
        a           = progress_bar_container, 
        id          = "progress_bar", 
        classes     = "absolute left-0 h-5 bg-yellow-300 z-0"
    )
    progress_text = jp.Span(
        a           = progress_bar_container, 
        text        = "0%", 
        id          = "progress_text", 
        classes     = "z-20"
    )

    def update(percentage):
        percentage = percentage * 100
        progress_bar.style = f"width: {percentage}%"
        progress_text.text = str(percentage) + "%"
    
    node_dict["webpage"].update_progress_bar = update

def setting_view(node_dict: dict):
    # =================== 설정 ===================
    setting_container = jp.Div(
        a           = node_dict["plane"], 
        id          = "setting_container", 
        classes     = "h-full flex flex-col border-box mx-5"
    )
    node_dict["setting_container"] = setting_container

    setting_controller = jp.Div(
        a           = setting_container, 
        id          = "setting_controller", 
        classes     = "flex flex-col items-end bg-gray-300 p-5 rounded-lg mt-10"
    )
    node_dict["setting_controller"] = setting_controller

    jp.Span(
        a           = setting_controller, 
        text        = "데이터 생성 설정", 
        classes     = "text-lg self-start font-bold mb-3"
    )

    # =================== 웹캠 정보 가져오기 ===================
    cam = cv2.VideoCapture(0)
    setting_container.width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    setting_container.height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    setting_container.cam_fps = cam.get(cv2.CAP_PROP_FPS)
    setting_container.user_fps = setting_container.cam_fps
    setting_container.clip_length = 3
    cam.release()

    # =================== 생성 시간 ===================
    def time_control(self, msg):
        time_value = float(self.value)
        user_fps = node_dict["setting_container"].user_fps
        node_dict["setting_control_frame_input"].value = time_value * user_fps  if user_fps > 0 else 0

        node_dict["setting_container"].time_length = time_value
        node_dict["setting_container"].frame_count = int(node_dict["setting_control_frame_input"].value)
    
    setting_control_time = jp.Div(
        a           = setting_controller, 
        id          = "setting_control_time", 
        classes     = "flex justify-around items-center"
    )
    jp.Span(
        a           = setting_control_time, 
        text        = "생성 길이(초)", 
        classes     = "text-base"
    )
    jp.Input(
        a           = setting_control_time, 
        id          = "setting_control_time_input", 
        placeholder = "초 단위로 입력", 
        classes     = "m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500"
    )
    node_dict["setting_control_time_input"] = setting_control_time.components[-1]
    node_dict["setting_control_time_input"].on("input", time_control)

    # =================== 생성 프레임 ===================
    def frame_control(self, msg):
        frame_value = float(self.value)
        user_fps = node_dict["setting_container"].user_fps
        node_dict["setting_control_time_input"].value = frame_value / user_fps if user_fps > 0 else 0

        node_dict["setting_container"].frame_count = frame_value
        node_dict["setting_container"].time_length = node_dict["setting_control_time_input"].value

    setting_control_frame = jp.Div(
        a           = setting_controller, 
        id          = "setting_control_frame", 
        classes     = "flex justify-around items-center"
    )
    jp.Span(
        a           = setting_control_frame, 
        text        = "생성 프레임 개수", 
        classes     = "text-base"
    )
    jp.Input(
        a           = setting_control_frame, 
        id          = "setting_control_frame_input", 
        placeholder = "프레임 개수를 입력", 
        classes     = "m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500"
    )
    node_dict["setting_control_frame_input"] = setting_control_frame.components[-1]
    node_dict["setting_control_frame_input"].on("input", frame_control)
    
    # =================== 3D pose 배치 크기 ===================
    def clip_control(self, msg):
        clip_value = int(self.value)
        clip_value = clip_value if clip_value > 0 else 1

        node_dict["setting_container"].clip_length = clip_value

    setting_control_clip = jp.Div(
        a           = setting_controller, 
        id          = "setting_control_clip", 
        classes     = "flex justify-around items-center"
    )
    jp.Span(
        a           = setting_control_clip, 
        text        = "3D pose 클립 크기", 
        classes     = "text-base"
    )
    jp.Input(
        a           = setting_control_clip, 
        id          = "setting_control_clip_input", 
        placeholder = "클립 크기 입력", 
        classes     = "m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500"
    )
    node_dict["setting_control_clip_input"] = setting_control_clip.components[-1]
    node_dict["setting_control_clip_input"].on("input", clip_control)

    # =================== 화면 캡처 주기 ===================
    def capture_interval_control(self, msg):
        user_fps = float(self.value)
        user_fps = user_fps if user_fps > 0 else 0
        if user_fps == 0:
            return

        # 생성 시간
        user_frame = node_dict["setting_control_frame_input"].value
        node_dict["setting_control_time_input"].value = user_frame / user_fps

        # 값 업데이트
        node_dict["setting_container"].time_length = node_dict["setting_control_frame_input"].value
        node_dict["setting_container"].user_fps = user_fps

    setting_control_capture_interval = jp.Div(
        a=setting_controller, 
        id="setting_control_capture_interval", 
        classes="flex justify-around items-center"
    )
    jp.Span(
        a           = setting_control_capture_interval, 
        text        = "화면 캡처 주기(fps)", 
        classes     = "text-base"
    )
    jp.Input(
        a           = setting_control_capture_interval, 
        id          = "setting_control_capture_interval_input", 
        placeholder = "fps 입력", 
        classes     = "m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500"
    )
    node_dict["setting_control_capture_interval_input"] = setting_control_capture_interval.components[-1]
    node_dict["setting_control_capture_interval_input"].on("input", capture_interval_control)
    node_dict["setting_control_capture_interval_input"].value = setting_container.cam_fps

    # =================== 웹캠 정보 ===================
    current_webcam_info_container = jp.Div(
        a           = setting_controller, 
        id          = "current_webcam_info_container", 
        classes     = "w-full flex justify-around items-center my-2"
    )
    jp.Span(
        a           = current_webcam_info_container, 
        text        = f"웹캠 fps : {setting_container.cam_fps}", 
        classes     = "text-sm text-purple-500 font-bold"
    )
    jp.Span(
        a           = current_webcam_info_container, 
        text        = f"이미지 크기 : {setting_container.width} x {setting_container.height}", 
        classes     = "text-sm text-purple-500 font-bold"
    )

    # =================== Progress Bar ===================
    progress_bar_view(setting_container, node_dict)

    # =================== 웹캠 및 수집 버튼 ===================
    def webcam_control(self, msg):
        def webcam_off():
            node_dict["webpage"].is_webcam_on = False
            self.set_class('bg-red-600')
            self.text = "웹캠 키기"

        def webcam_on():
            node_dict["webpage"].is_webcam_on = True
            self.text = "웹캠 끄기"
            self.set_class('bg-purple-400')

            jp.run_task(update_webcam(node_dict))

        if msg is None:
            node_dict["webpage"].webcam_off = webcam_off
            node_dict["webpage"].webcam_on = webcam_on
        else:
            if node_dict["webpage"].is_webcam_on:
                webcam_off()
            else:
                webcam_on()

    def collection_control(self, msg):
        def collecting_off():
            node_dict["webpage"].update_progress_bar(0)

            node_dict["webpage"].is_collection_on = False
            self.set_class('bg-yellow-600')
            self.text = "데이터 수집하기"
        
        def collecting_on():
            if not(node_dict["webpage"].is_webcam_on):
                node_dict["webpage"].webcam_on()

            node_dict["webpage"].update_progress_bar(0)

            self.text = "수집 종료하기"
            self.set_class('bg-pink-600')
            
            collection_box = {
                "date"          : datetime.now().strftime("%Y-%m-%d(%H:%M:%S)"),
                "hash"          : md5(str(time.time()).encode("utf-8")).hexdigest(),
                "img_width"     : node_dict["setting_container"].width,
                "img_height"    : node_dict["setting_container"].height,
                "frame_count"   : node_dict["setting_container"].frame_count,
                "clip_length"   : node_dict["setting_container"].clip_length,
                "fps"           : node_dict["setting_container"].user_fps,
                "time_length"   : node_dict["setting_container"].time_length,
                "datas"         : []
            }

            node_dict["webpage"].collected_data.append(collection_box)

            node_dict["webpage"].is_collection_on = True

        if msg is None:
            node_dict["webpage"].collecting_off = collecting_off
            node_dict["webpage"].collecting_on = collecting_on
        else:
            if node_dict["webpage"].is_collection_on:
                collecting_off()
            else:
                collecting_on()

    core_button_container = jp.Div(
        a=setting_container, 
        id="core_button_container", 
        classes="flex justify-center mt-5"
    )
    node_dict["webpage"].is_webcam_on = False
    node_dict["webpage"].is_collection_on = False

    jp.Button(
        a           = core_button_container, 
        id          = "webcam_control_btn", 
        text        = "웹캠 키기", 
        classes     = "w-32 m-2 bg-red-600 hover:bg-red-100 hover:text-black text-white font-bold py-3 px-4 rounded-lg"
    )
    jp.Button(
        a           = core_button_container, 
        id          = "data_collection_control_btn", 
        text        = "데이터 수집하기", 
        classes     = "w-35 m-2 bg-yellow-600 hover:bg-red-100 hover:text-black text-white font-bold py-3 px-4 rounded-lg"
    )

    # 초기화
    webcam_control(core_button_container.components[0], None)
    collection_control(core_button_container.components[1], None)

    core_button_container.components[0].on("click", webcam_control)
    core_button_container.components[1].on("click", collection_control)

def post_process_keypoints(parsed_data:dict):
    frame_count = parsed_data["frame_count"]
    datas = parsed_data["datas"]

    processed_keypoints_3d = [[] for _ in range(frame_count)]
    processed_scores = [[] for _ in range(frame_count)]

    for idx, data in enumerate(datas):
        keypoints_3d = data["pose3d_output"]
        keypoints_3d = np.array(keypoints_3d) # (17, 3, T)

        scores = data["keypoints_scores"]
        scores = np.array(scores) # (T, 17)

        clip_length = keypoints_3d.shape[2]

        valid_data_start = max(0, idx - clip_length + 1)
        valid_data_end = idx + 1
        valid_data_size = valid_data_end - valid_data_start

        for valid_idx in range(valid_data_start, valid_data_end, 1):
            data_idx = clip_length - valid_data_size + (valid_idx - valid_data_start)

            processed_keypoints_3d[valid_idx].append(keypoints_3d[..., data_idx])
            processed_scores[valid_idx].append(scores[data_idx])

    for idx, (target_kp_3d, target_score) in enumerate(zip(processed_keypoints_3d, processed_scores)):
        target_kp_3d = np.stack(target_kp_3d) # (len, 17, 3)
        target_kp_3d = np.mean(target_kp_3d, axis=0) # (17, 3)
        
        target_score = np.stack(target_score) # (len, 17)
        target_score = np.mean(target_score, axis=0) # (17,)

        parsed_data["datas"][idx]["pose3d_output"] = target_kp_3d.tolist()
        parsed_data["datas"][idx]["keypoints_scores"] = target_score.tolist()
    
    return parsed_data

def download_item(node_dict: dict, collected_datas):
    """
    collection_box = {
        "date": datetime.now().strftime("%Y-%m-%d(%H:%M)"),
        "hash": md5(str(time.time()).encode("utf-8")).hexdigest(),
        "img_width": self.a.a.width,
        "img_height": self.a.a.height,
        "frame_count": self.a.a.frame_count,
        "clip_length": self.a.a.clip_length,
        "fps": self.a.a.user_fps,
        "time_length": self.a.a.time_length,
        "datas": []
    }

    # In collection_box["datas"]
    collected_data = {
        "index": len(webpage.collected_data[-1]["datas"]),
        "img_jpeg": img_jpeg,
        "pose3d_output": motion_world,
        "keypoints_scores": keypoints_scores
    }
    """
    item_container  = jp.Div(a=node_dict["download_list_container"], classes="h-15 flex p-3 items-center bg-green-100 mb-3")
    date            = collected_datas["date"]
    frame_count     = int(collected_datas["frame_count"])
    clip_length     = collected_datas["clip_length"]
    item_name       = f"{date}_fc{frame_count}_cl{clip_length}"

    jp.Span(a=item_container, text=item_name, classes="text-base mr-5")
    download_btn = jp.A(a=item_container, text="💾", classes="text-2xl cursor-pointer")

    hash = collected_datas["hash"]
    
    selected_data = None
    for data in node_dict["webpage"].collected_data:
        if data["hash"] == hash:
            selected_data = data
            break

    selected_data = post_process_keypoints(selected_data)
    
    json_file = item_name + ".json"
    with open(json_file, "w") as f:
        json.dump(selected_data, f)
    
    download_btn.href=f"/static/{json_file}"
    download_btn.download=json_file

def download_view(node_dict: dict):
    download_container = jp.Div(
        a           = node_dict["plane"], 
        id          = "download_container", 
        classes     = "h-full flex flex-col items-center border-box bg-white border border-green-400 p-5"
    )
    jp.Span(
        a           = download_container, 
        text        = "저장된 데이터 목록", 
        classes     = "text-lg font-bold mb-1"
    )

    download_list_container = jp.Div(
        a           = download_container, 
        id          = "download_list_container", 
        classes     = "h-full flex flex-col"
    )
    node_dict["download_list_container"] = download_list_container

    def add_collected_item(collected_datas):
        download_item(node_dict, collected_datas)
    
    node_dict["webpage"].add_collected_item = add_collected_item

def page_ready(self, msg):
    jp.run_task(self.run_javascript("""
        const svg = document.querySelector('#pose3d_figures svg');
        if (svg) {
            svg.style.width = '100%';
            svg.style.height = '100%';
        }
    """))

def data_collection_page():
    wp = jp.WebPage()
    wp.on('page_ready', page_ready)
    wp.collected_data = []
    wp.node_dict = dict()
    wp.node_dict["webpage"] = wp

    plane = jp.Div(
        a           = wp, 
        id          = "plane", 
        classes     = "h-screen w-screen flex border-box justify-center py-3"
    )
    wp.node_dict["plane"] = plane
    
    result_view(node_dict=wp.node_dict)
    setting_view(node_dict=wp.node_dict)
    download_view(node_dict=wp.node_dict)

    return wp

if __name__ == "__main__":
    jp.justpy()