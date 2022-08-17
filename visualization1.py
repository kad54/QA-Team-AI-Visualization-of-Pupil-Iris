import math
import numpy as np
import random
import cv2

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
    thickness = 1
    line = cv2.LINE_AA

    rectangle = cv2.rectangle(frame, start_point_iris, end_point_iris, color, thickness, line)
    rectangle = cv2.rectangle(rectangle, start_point_pupil, end_point_pupil, color, thickness, line)

    return rectangle