import math
import numpy as np
import random
import cv2
from return_coordinates import return_circle_coordinates, return_rectangle_coordinates

length = 100
x = 100
y = 100

def draw_line(img, x1, y1, x2, y2, color=(0,0,255), thickness=1):
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_ruler_horizontal(img, x1, y1, x2, y2, color=(0,0,255), thickness=1):
    draw_line(img, x1, y1, x2, y2, color, thickness)
    for i in range(x1, x2, 20):
        draw_line(img, i, y1, i, y1+5, color, thickness)

def draw_ruler_vertical(img, x1, y1, x2, y2, color=(0,0,255), thickness=1):
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    for i in range(y1, y2, 20):
        cv2.line(img, (x1, i), (x1+5, i), color, thickness)

def add_circles(frame, prediction):
    """
    Adds rectangles from YOLO prediction to a given frame
    Input: frame with [x_dim, y_dim, channels], prediction dictionary created by YOLO predict function
    Output: frame with drawn rectangles from predictions on it, [x_dim, y_dim, channels]
    """

    center_iris, radius_iris = return_circle_coordinates(prediction["iris"])
    center_pupil, radius_pupil = return_circle_coordinates(
        prediction["pupil"]
    )
    color = (210, 210, 210)
    thickness = 1
    line = cv2.LINE_AA

    circle = cv2.circle(frame, center_iris, radius_iris, color, thickness, line)
    circle = cv2.circle(circle, center_pupil, radius_pupil, color, thickness, line)

    color2 = (136,173,210)

    for i in range(0, 360, 90):
        x = int(center_iris[0] + radius_iris * math.cos(math.radians(i)))
        y = int(center_iris[1] + radius_iris * math.sin(math.radians(i)))
        cv2.line(circle, center_iris, (x, y), color2, thickness, line)  

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 550)
    bottomLeftCornerOfText2 = (20, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    r = round((radius_pupil/6), 2)

    cv2.putText(circle, 'Diameter: ' + str(r) + ' mm',
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)
    
    brightness = (random.randint(75, 100))

    cv2.putText(circle, 'Brightness: ' + str(brightness) + ' %',
        bottomLeftCornerOfText2,
        font,
        fontScale,
        fontColor,
        lineType)

    return circle


def add_rectangles(frame, prediction):
    """
    Adds rectangles from YOLO prediction to a given frame
    Input: frame with [x_dim, y_dim, channels], prediction dictionary created by YOLO predict function
    Output: frame with drawn rectangles from predictions on it, [x_dim, y_dim, channels]
    """
    start_point_iris, end_point_iris = return_rectangle_coordinates(prediction["iris"])
    #start_point_pupil, end_point_pupil = return_rectangle_coordinates(
        #prediction["pupil"]
    #)
    x1_iris, y1_iris = start_point_iris
    x2_iris, y2_iris = end_point_iris
    color = (210, 210, 210)
    thickness = 1
    r = 5
    d = 20
    
    frame_copy = np.copy(frame)
    # Bottom left
    cv2.line(frame_copy, (x1_iris + r, y1_iris), (x1_iris + r + d, y1_iris), color, thickness)
    cv2.line(frame_copy, (x1_iris, y1_iris - r), (x1_iris, y1_iris - r - d), color, thickness)
    cv2.ellipse(frame_copy, (x1_iris + r, y1_iris - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(frame_copy, (x2_iris - r, y1_iris), (x2_iris - r - d, y1_iris), color, thickness)
    cv2.line(frame_copy, (x2_iris, y1_iris - r), (x2_iris, y1_iris - r - d), color, thickness)
    cv2.ellipse(frame_copy, (x2_iris - r, y1_iris - r), (r, r), 0, 0, 90, color, thickness)

    # Top left
    cv2.line(frame_copy, (x1_iris + r, y2_iris), (x1_iris + r + d, y2_iris), color, thickness)
    cv2.line(frame_copy, (x1_iris, y2_iris + r), (x1_iris, y2_iris + r + d), color, thickness)
    cv2.ellipse(frame_copy, (x1_iris + r, y2_iris + r), (r, r), 90, 90, 180, color, thickness)
    # Top right
    cv2.line(frame_copy, (x2_iris - r, y2_iris), (x2_iris - r - d, y2_iris), color, thickness)
    cv2.line(frame_copy, (x2_iris, y2_iris + r), (x2_iris, y2_iris + r + d), color, thickness)
    cv2.ellipse(frame_copy, (x2_iris - r, y2_iris + r), (r, r), 270, 0, 90, color, thickness)
    
    return frame_copy