import cv2
import numpy as np
import imutils 
import pytesseract
import io

def bottom_part(img):
    midgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ww, hw = midgg.shape[::-1]
    img=img[75:hw,1:ww]
    BLUnit = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    BLTen = cv2.blur(BLUnit, (3, 3))
    BLUnitG = BLUnit
    sobelxBLUnit = cv2.Sobel(BLTen,cv2.CV_8U,1,0,ksize=3)
    ret2,threshold_imgBLUnit = cv2.threshold(sobelxBLUnit,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 1))
    morph_img_thresholdBLUnit = threshold_imgBLUnit.copy()
    cv2.morphologyEx(src=threshold_imgBLUnit, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_thresholdBLUnit)
    # Creating kernel
    kernel = np.ones((5, 5), np.uint8)
    # Using cv2.erode() method 
    morph_img_thresholdBLUnit = cv2.erode(morph_img_thresholdBLUnit, kernel) 

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((15,15), np.uint8)
    morph_img_thresholdBLUnit1 = cv2.dilate(morph_img_thresholdBLUnit, kernel, iterations=1)

    contoursBLUnit = cv2.findContours(morph_img_thresholdBLUnit1.copy(),cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    contoursBLUnit = imutils.grab_contours(contoursBLUnit)
    contoursBLUnit = sorted(contoursBLUnit,key=cv2.contourArea, reverse = True)[:1]
    screenCnt = None
    for c in contoursBLUnit:
        # approximate the contour
        x,y,w,h = cv2.boundingRect(c) 
        rect = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),0)
        """ cv2.imshow("block_lp", rect)
        cv2.waitKey(0) """

    last = img[y:y+h, x:x+w]
    last_gary = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
    ret, last_bw = cv2.threshold(last_gary, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    roi = cv2.bitwise_not(last_bw)
    """ cv2.imshow("block_lp", roi)
    cv2.waitKey(0) """
    
    config="""-l tha --psm 8 -c tessedit_char_blacklist=0123456789?!@#$%^&-_+=}{][)(\/><|:;,~*`.\\"\\'"""
    province = pytesseract.image_to_string(roi, config=config)
    province = province.replace("\n", " ").replace(" ", "")
    print("province : ", province)

    return str(province)

