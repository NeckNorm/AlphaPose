import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy
import numpy as np
import justpy as jp
import cv2
import base64

from typing import Union


def pose3d_visualize(
    ax                  : matplotlib.axes.Axes, 
    motion              : numpy.ndarray, 
    scores              : numpy.ndarray, 
    elivation           : int, 
    angle               : int, 
    keypoints_threshold : float=0.7
) -> None:
    """
    Params
    ---
    ax : Figure of 3D plot
    motion : 3D keypoints, shape = (17, 3)
    scores : Keypoints scores, shape = (17)
    elivation : Elevation angle
    angle : Rotation angle
    keypoints_threshold : Threshold for keypoints scores

    Returns
    ---
    None
    """
    joint_pairs         = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left    = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right   = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

    color_mid   = "#fc0313" # Red
    color_left  = "#02315E" # Blue
    color_right = "#19a303" # Green

    ax.set_xlim(-512, 0)
    ax.set_ylim(-256, 256)
    ax.set_zlim(-512, 0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=elivation, azim=angle)

    plt.tick_params(
        left        = False, 
        right       = False, 
        labelleft   = False, 
        labelbottom = False, 
        bottom      = False
    )

    for joint_pair in joint_pairs:
        # 두 Keypoint 중 하나라도 threshold 미만이면 시각화 하지 않음
        if (scores[joint_pair[0]] < keypoints_threshold) or (scores[joint_pair[1]] < keypoints_threshold):
            continue

        xs = np.array(motion[joint_pair, 0])
        ys = np.array(motion[joint_pair, 1])
        zs = np.array(motion[joint_pair, 2])

        def plot_joint_pair(color):
            ax.plot(
                -xs, -zs, -ys, 
                color=color, 
                lw=3, 
                marker='o', 
                markerfacecolor='w', 
                markersize=3, 
                markeredgewidth=2
            )

        if joint_pair in joint_pairs_left:
            plot_joint_pair(color=color_left)
        elif joint_pair in joint_pairs_right:
            plot_joint_pair(color=color_right)
        else:
            plot_joint_pair(color=color_mid)

def screen_update(screen: jp.Div, image: Union[str, cv2.typing.MatLike]):
    if isinstance(image, cv2.typing.MatLike):
        _, jpeg = cv2.imencode('.jpg', image)
        img_jpeg = base64.b64encode(jpeg)
        img_as_text = img_jpeg.decode('utf-8')
        screen.src = f'data:image/jpeg;base64,{img_as_text}'
    elif isinstance(image, str):
        screen.src = image
    else:
        raise ValueError("Invalid image type")
