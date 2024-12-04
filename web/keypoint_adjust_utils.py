import numpy as np
import torch

def adjust_head_pose(kps_3d, kps_2d):
    """
    kp_3d : (17, 3, T)
    kp_2d : (T, 17, 2)
    """
    clip_len = kps_3d.shape[2]
    for i in range(clip_len):
        kp_3d = kps_3d[...,i]
        kp_2d = kps_2d[i]

        vector_head_2d = kp_2d[8] - kp_2d[10]
        vector_head_2d = vector_head_2d / np.linalg.norm(vector_head_2d)

        vector_head_3d = kp_3d[8] - kp_3d[10]
        vector_head_3d = vector_head_3d / np.linalg.norm(vector_head_3d)
        
        angle = np.arccos(np.dot(vector_head_3d[:2], vector_head_2d))

        if vector_head_2d[0] > vector_head_3d[0]:
            angle = -angle

        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

        adjusted_coords = (kp_3d[8:11] - kp_3d[10:11])[...,:2] @ rotation_matrix.T + kp_3d[10:11][...,:2]
        kp_3d[8:11][...,:2] = adjusted_coords

    return kps_3d

def adjust_neck_pose(kps_3d):
    """
    kp_3d : (17, 3, T)
    """
    clip_len = kps_3d.shape[2]
    for i in range(clip_len):
        kp_3d = kps_3d[...,i]

        vector_head_3d = kp_3d[8] - kp_3d[10]
        vector_head_3d = vector_head_3d / np.linalg.norm(vector_head_3d)

        if vector_head_3d[2] > 0:
            adjusted_coords = kp_3d - kp_3d[10:11] # 머리가 원점이 되도록
            adjusted_coords[...,2] = -adjusted_coords[...,2] # z축 반전
            adjusted_coords = adjusted_coords + kp_3d[10:11] # 원래 좌표로 이동
            
            kps_3d[...,i] = adjusted_coords

    return kps_3d