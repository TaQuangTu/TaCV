from typing import Union

import cv2
import numpy as np

def draw_points(image,points: Union[list,tuple,np.ndarray],circular=True,color=(255,0,0),thickness=2):
    '''
    :param image:
    :param points: Must be in shape of N x 2, where N is the number of points
    :param circular:
    :param color:
    :param thickness:
    :return:
    '''
    points = [(int(point[0]),int(point[1])) for point in points]
    N = len(points)
    if not circular:
        N = N - 1
    for i in range(N):
        cv2.line(image,points[i],points[(i+1)%N],color,thickness)