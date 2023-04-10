import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 100, 200)
    return edged

def find_largest_rectangle_contour(contours):
    largest_area = 0
    largest_rect = None

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        area = cv2.contourArea(box)
        if area > largest_area:
            largest_area = area
            largest_rect = box

    return largest_rect

def perspective(image):
    preprocessed_image = preprocess_image(image)

    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_rect = find_largest_rectangle_contour(contours)

    src_points = order_points(largest_rect).astype("float32")

    output_width = 200
    output_height = 40

    dst_points = np.float32([
    [0, 0],
    [output_width, 0],
    [output_width, output_height],
    [0, output_height]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(image, matrix, (output_width, output_height))

    cv2.imshow('output_image.jpg', result)
    cv2.waitKey(0)

    return result


