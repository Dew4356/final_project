import numpy as np
import cv2

# map colour names to HSV ranges
color_list = [
    ['ดำ', [0, 0, 0], [180, 255, 30]],
    ['เทา', [180, 18, 230], [0, 0, 40]],
    ['แดง', [160, 50, 50], [180, 255, 255]],
    ['เหลือง', [15, 50, 70], [30, 250, 250]],
    ['เขียว', [36, 50, 70], [89, 255, 255]],
    ['ส้ม', [5, 50, 50], [15, 255, 255]],
    ['น้ำเงิน', [90, 50, 70], [128, 255, 255]],
    ['ม่วง', [129, 50, 70], [158, 255, 255]],
    ['ขาว', [0, 0, 231], [180, 18, 255]],
    ['ฟ้า', [80, 50, 70], [90, 250, 250]],
    ['ชมพู', [170, 50, 70], [180, 255, 255]],
    ['น้ำตาล', [10, 100, 20], [20, 255, 200]],
]

def detect_color(hsv_image):
    hsv = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)
    color_found = 'undefined'
    max_count = 0

    """ cv2.imshow("mask", hsv)
    cv2.waitKey(0) """

    for color_name, lower_val, upper_val in color_list:
        # threshold the HSV image - any matching color will show up as white
        mask = cv2.inRange(hsv, np.array(lower_val), np.array(upper_val))
        
        """ cv2.imshow("mask", mask)
        cv2.waitKey(0) """
        
        """ res = cv2.bitwise_and(hsv_image, hsv, mask= mask) """
        
        # count white pixels on mask
        count = np.sum(mask)
        print('count', count)

        if count > max_count:
            color_found = color_name
            max_count = count

    return color_found