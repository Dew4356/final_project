import numpy as np
import cv2
import pytesseract
import re

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

    """ width = dilation.shape[1]
    split_point = int(width * 0.4)  # Adjust the split point based on your license plate format
    first_part = dilation[:, :split_point]
    last_4_digits = dilation[:, split_point:] """

    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

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

        special_chars = "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~๐๑๒๓๔๕๖๗๘๙฿\"\"\'\'" # " ,, '
        config = f"-l tha --psm 6 -c tessedit_char_blacklist={special_chars}"
        
        text = pytesseract.image_to_string(roi, config=config)        
        text = text.replace("\n", "")

        thai_vowels_pattern = r"[\u0E30-\u0E3A\u0E40-\u0E4f]"
        text = re.sub(thai_vowels_pattern, '', text).replace(" ", "")

        text = re.sub(r'(?<=\D)(\D{1,4})$', '', text)

        print(text)
        
        number.append(text)
        lens = len(number)
        ress=""

        for i in range(lens):
            ress = ress + number[i]

    print("lp_number : " + ress)

    return str(ress).replace("\f", "")
