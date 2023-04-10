import torch
from PIL import Image
import cv2
import numpy as np
from province2 import getProvince
from toppart import top_part
from bottompart import bottom_part
from carpart import detect_color

# Model license plate
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/best.pt')
model.max_det = 1

# Model car
model2 = torch.hub.load('ultralytics/yolov5', 'yolov5m')
model2.classes = [2]
model2.max_det = 1

def resize_img(img):
    target_width = 400
    aspect_ratio = img.shape[1] / img.shape[0]
    target_height = int(target_width / aspect_ratio)
    resized = cv2.resize(img, (target_width, target_height), interpolation = cv2.INTER_CUBIC)

    return resized

def upload_image(img):
    predict = model(img)

    predict = predict.pandas().xyxy[0][predict.pandas().xyxy[0].confidence >= 0.7]

    bboxes = (list(zip(predict.xmin.values, 
                predict.ymin.values, 
                predict.xmax.values, 
                predict.ymax.values)))

    for bbox in bboxes :
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        plate_roi = img[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]

    new_img = resize_img(plate_roi)
        
    top = top_part(new_img)
    btm = bottom_part(new_img)
    province = getProvince(btm)

    return top, province

img = cv2.imread('test_img/test.jpg')
upload_image(img)