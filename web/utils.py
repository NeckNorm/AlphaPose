import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy
import numpy as np
import justpy as jp
import cv2
import base64

from typing import Union

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


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
    joint_pairs = [[0,7], [7,8], [8,9], [9,10],                         # Center
                   [8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], # Right
                   [8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]] # Left
    
    joint_pairs_left    = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right   = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

    color_mid   = "#fc0313" # Red
    color_left  = "#02315E" # Blue
    color_right = "#19a303" # Green

    ax.set_xlim(0, 512)
    ax.set_ylim(-256, 256)
    ax.set_zlim(-512, 0)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    ax.view_init(elev=elivation, azim=angle)

    plt.tick_params(
        left        = False, 
        right       = False, 
        labelleft   = False, 
        labelbottom = False, 
        bottom      = False
    )

    # center = motion[8]
    # cx, cy, cz = -center
    # # X axis
    # ax.arrow3D(
    #     cx,cz,cy,
    #     512,0,0,
    #     mutation_scale=20,
    #     ec ='red',
    #     fc='red'
    # )

    # # Z axis
    # ax.arrow3D(
    #     cx,cz,cy,
    #     0,256,0,
    #     mutation_scale=20,
    #     ec ='blue',
    #     fc='blue'
    # )

    # # Y axis
    # ax.arrow3D(
    #     cx,cz,cy,
    #     0,0,512,
    #     mutation_scale=20,
    #     ec ='green',
    #     fc='green'
    # )

    for joint_pair in joint_pairs:
        # 두 Keypoint 중 하나라도 threshold 미만이면 시각화 하지 않음
        if (scores[joint_pair[0]] < keypoints_threshold) or (scores[joint_pair[1]] < keypoints_threshold):
            continue

        xs = np.array(motion[joint_pair, 0])
        ys = np.array(motion[joint_pair, 1])
        zs = np.array(motion[joint_pair, 2])

        def plot_joint_pair(color):
            ax.plot(
                xs, -zs, -ys, 
                color=color, 
                lw=1, 
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

        return img_as_text
    elif isinstance(image, str):
        screen.src = f"data:image/jpeg;base64,{image}"

        return image
    else:
        raise ValueError("Invalid image type")
