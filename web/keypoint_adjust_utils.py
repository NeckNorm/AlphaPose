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

        vector_head_2d = kp_2d[10] - kp_2d[8]
        vector_head_2d = vector_head_2d / np.linalg.norm(vector_head_2d)

        vector_head_3d = kp_3d[10] - kp_3d[8]
        vector_head_3d = vector_head_3d / np.linalg.norm(vector_head_3d)

        angle = np.arccos(np.dot(vector_head_2d, vector_head_3d[:2]))

        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

        adjusted_coords = (kp_3d[8:10] - kp_3d[10:11])[...,:2] @ rotation_matrix + kp_3d[10:11][...,:2]
        kp_3d[8:10][...,:2] = adjusted_coords

    return kps_3d

def adjust_neck_depth(kps_3d):
    kps_3d = kps_3d.cpu().numpy()
    for i in range(kps_3d.shape[2]):
        kp_3d = np.array(kps_3d[...,i])
        line = kp_3d[11][[0,2]], kp_3d[14][[0,2]]
        dx = line[1][0] - line[0][0]
        dy = line[1][1] - line[0][1]
        slope = dy / dx if dx != 0 else 0

        line_function = lambda x: slope * (x - line[0][0]) + line[0][1]

        neck_diff = line_function(kp_3d[8][0]) - kp_3d[8][2]

        if neck_diff < 0:
            kp_3d[8:11,2] = kp_3d[8:11,2] + 2*neck_diff
        
        kps_3d[...,i] = kp_3d

    return torch.FloatTensor(kps_3d)