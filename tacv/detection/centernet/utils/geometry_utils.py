import numpy as np
import cv2
from tacv.visual import draw_points


def is_out_side_image(x, y, w, h):
    """
    Check 2D location is outside image of shape (w,h)
    :param x:
    :param y:
    :param w:
    :param h:
    :return: True if outside image, otherwise returns False
    """
    if x < 0 or x >= w or y < 0 or y >= h:
        return True
    return False


def rotation_of_line(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    return np.arctan2(delta_x, delta_y)


def rotation_matrix(rotation):
    cos = np.cos(rotation)
    sin = np.sin(rotation)

    return np.array((
        [cos, -sin],
        [sin, cos])
    )


def rotate_points_around_another(p, other, rot):
    '''
    :param p: N x 2, 1st col = Xs, 2nd col = Ys
    :param other: another point. np array of 2.1
    :return: rotation in radian
    '''
    other_numpy = np.array(other).reshape(2, 1)
    point_in_other_coor_sys = np.array(p).T - other_numpy
    rot_mat = rotation_matrix(rot)
    rotated_points = rot_mat @ point_in_other_coor_sys
    rotated_points += other_numpy
    return rotated_points.T


def get_perspective_transform_matrix(src, det):
    matrix = cv2.getPerspectiveTransform(src, det)
    return matrix


def warp_image_with_perspective_transform(image, mat, dsize=(720, 1080), inplace=False):
    if inplace:
        cv2.warpPerspective(image, mat)
        return image
    else:
        new_image = cv2.warpPerspective(image.copy(), mat, dsize=dsize)
        return new_image


def get_contours(image):
    canny1 = 255 - cv2.Canny(image, threshold1=50, threshold2=200)
    # canny2 = cv2.Canny(image, threshold1=100, threshold2=200)
    # sobel = cv2.Sobel(image,ddepth=cv2.CV_8U,dy=1,dx=1,ksize=5)
    # cv2.imshow("canny1",canny1)
    # cv2.imshow("canny2", canny2)
    # cv2.imshow("sobel", sobel)
    # cv2.waitKey(0)

    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    thresh = np.where(thresh<canny1,thresh,canny1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_top_k_areas(contours, k=3, take_max=True):
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=take_max == True)
    return contours[:k]


def detect_corners_and_do_perspective_transform(image):
    contours = get_contours(image)
    ## only draw contour that have big areas
    imx = image.shape[0]
    imy = image.shape[1]
    image_area = imx * imy
    lp_area = image_area / 10

    ## Get only rectangles given exceeding area
    contours = get_top_k_areas(list(contours), k=5)
    approx_polygons = [cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True) for cnt in contours]

    for approx, contour in zip(approx_polygons, contours):
        if len(approx) == 4 and cv2.contourArea(contour) > lp_area:
            tmp_img = image.copy()
            cv2.drawContours(tmp_img, [approx], 0, (0, 255, 255), 6)
            cv2.imshow("contour", tmp_img)
            cv2.waitKey(0)
