import math
import numpy as np
import random
import cv2
from return_coordinates import return_circle_coordinates, return_rectangle_coordinates


def add_rectangles(frame, prediction):
    """
    Adds rectangles from YOLO prediction to a given frame
    Input: frame with [x_dim, y_dim, channels], prediction dictionary created by YOLO predict function
    Output: frame with drawn rectangles from predictions on it, [x_dim, y_dim, channels]
    """

    start_point_iris, end_point_iris = return_rectangle_coordinates(prediction["iris"])
    start_point_pupil, end_point_pupil = return_rectangle_coordinates(
        prediction["pupil"]
    )
    color = (210, 210, 210)

    thickness = 2
  
    rectangle = cv2.rectangle(frame, start_point_pupil, end_point_pupil, color, thickness-1)

    return rectangle
def add_circles(frame, prediction):
    """
    Adds rectangles from YOLO prediction to a given frame
    Input: frame with [x_dim, y_dim, channels], prediction dictionary created by YOLO predict function
    Output: frame with drawn rectangles from predictions on it, [x_dim, y_dim, channels]
    """
    start_point_iris, end_point_iris = return_rectangle_coordinates(prediction["iris"])
    start_point_pupil, end_point_pupil = return_rectangle_coordinates(
        prediction["pupil"]
    )
    x1_iris, y1_iris = start_point_iris
    x2_iris, y2_iris = end_point_iris
    
    center_iris, radius_iris = return_circle_coordinates(prediction["iris"])
    center_pupil, radius_pupil = return_circle_coordinates(
        prediction["pupil"]
    )
    (x1, y1) = center_pupil

    color = (210, 210, 210)
    thickness = 1
    line = cv2.LINE_AA
    circle = cv2.circle(frame, center_pupil, radius_pupil, color, thickness)
    circle = cv2.line(frame, (x1, y2_iris), (x1, y1_iris), (0, 255, 0), 2)
    circle = cv2.line(frame, (x1_iris, y1), (x2_iris, y1), (0, 255, 0), 2)
    circle = cv2.circle(frame, center_pupil, radius_pupil - 19, color, thickness+5)
    # circle = cv2.circle(frame, center_pupil, radius_pupil, color, thickness)
    # circle = cv2.line(frame, (x1, 640), (x1, 0), (0, 255, 0), thickness + 2)
    # circle = cv2.line(frame, (0, y1), (640, y1), (0, 255, 0), 2)
    # circle = cv2.circle(circle, center_iris, radius_iris, color, thickness, line)
