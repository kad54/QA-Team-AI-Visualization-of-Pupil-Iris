import math
import numpy as np
import random
import cv2

def return_circle_coordinates(prediction):
    """
    This function takes the dictionary returnes by predict function of YOLO and changes it into format
    useful for cv2.circle
    Input: dictionary with prob, x, y, w, h floats
    Outputs: tuple of two tuples (start point x, y), (end point x, y)
    """

    x = prediction["x"]
    y = prediction["y"]
    w = prediction["w"]
    h = prediction["h"]

    center_x = x
    center_y = y
    radius = w / 2

    return (int(center_x), int(center_y)), int(radius)

def return_rectangle_coordinates(prediction):
    """
    This function takes the dictionary returnes by predict function of YOLO and changes it into format
    useful for cv2.rectangle
    Input: dictionary with prob, x, y, w, h floats
    Outputs: tuple of two tuples (start point x, y), (end point x, y)
    """

    x = prediction["x"]
    y = prediction["y"]
    w = prediction["w"]
    h = prediction["h"]

    start_x = x - w / 2
    start_y = y + h / 2
    end_x = x + w / 2
    end_y = y - h / 2

    return (int(start_x), int(start_y)), (int(end_x), int(end_y))