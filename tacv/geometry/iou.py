from typing import Union

import numpy
import numpy as np
from shapely.geometry import Polygon

def iou_2d(points:Union[list,tuple,numpy.ndarray],other_points:Union[list,tuple,numpy.ndarray]):
    """
    Calculate 2D iou of two polygons made from 2 sets of 2d points
    :param points: in shap of N x 2
    :param other_points: in shap of N x 2
    :return: iou 2d ranging from 0 to 1
    """
    def calculate_iou(box_1, box_2):
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou
    return calculate_iou(points,other_points)

if __name__=="__main__":
    points = [[0,0],[10,10],[0,10]]
    other_points = [[0, 20], [10, 10], [0, 0]]
    print(iou_2d(points,other_points))