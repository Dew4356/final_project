import numpy as np
import io
import json
import cv2
import pytesseract

def top_part(img):
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load the face cascade detector,
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # detect faces in the image
    gray = cv2.medianBlur(gray, 3)

    # construct a list of bounding boxes from the detection
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # update the data dictionary with the faces detected
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # return the data dictionary as a JSON response

    number = []

    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = gray.shape
        
        # if height of box is not a quarter of total height then skip
        if height / float(h) > 6: continue
        ratio = h / float(w)
        # if height to width ratio is less than 1.25 skip
        if ratio < 1.25: continue
        area = h * w
        # if width is not more than 25 pixels skip
        if width / float(w) > 25: continue
        # if area is less than 100 pixels skip
        if area < 100: continue
        # draw the rectangle
        rect = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),0)
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 3)

        config="""-l tha --psm 6 -c tessedit_char_blacklist=?!@#$%^&-_+=}{][)(\/><|:;,~*`.\\"\\' 
                \u0e30\u0e31\u0e32\u0e33\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39\u0e3a\u0e3f
                \u0e40\u0e41\u0e42\u0e43\u0e44\u0e45\u0e46\u0e47\u0e48\u0e49
                \u0e4a\u0e4b\u0e4c\u0e4d\u0e4e\u0e4f 
                \u0e50\u0e51\u0e52\u0e53\u0e54\u0e55\u0e56\u0e57\u0e58\u0e59\u0e5a\u0e5b""" #  " , '
        
        user_patterns_file = 'recognized.txt'
        custom_config = f'-c user_patterns_file={user_patterns_file} --psm 6 -l tha'
        
        text = pytesseract.image_to_string(roi, config=config)

        print(text)
        
        text = text.replace("\n", "")
        number.append(text)
        lens = len(number)
        ress=""

        for i in range(lens):
            ress = ress + number[i]

    return str(ress)
