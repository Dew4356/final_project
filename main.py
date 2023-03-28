import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import json
from PIL import Image
import cv2
from starlette.responses import Response, FileResponse
import numpy as np
from province2 import getProvince
from toppart import top_part
from bottompart import bottom_part
from carpart import detect_color
from fastapi.middleware.cors import CORSMiddleware
import logging
import base64

#create your API
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Enable CORS for the Angular app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model license plate
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/best.pt')
model.max_det = 1

# Model car
model2 = torch.hub.load('ultralytics/yolov5', 'yolov5m')
model2.classes = [2]
""" model2.max_det = 1 """
    
@app.post("/lp-detection/")
async def upload_image(request: Request, image: UploadFile = File()):
    contents = await image.read()

    # Read the uploaded image
    nparr = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    predict = model(img)

    result = json.loads(predict.pandas().xyxy[0].to_json(orient="records"))

    x1 = int(result[0]['xmin'])
    y1 = int(result[0]['ymin'])
    x2 = int(result[0]['xmax'])
    y2 = int(result[0]['ymax'])

    if(result):
        image_new = img[y1: y2, x1: x2]
        top_resize = cv2.resize(image_new, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        """ scale_percent = 60 # percent of original size
        width = int(image_new.shape[1] * scale_percent / 100)
        height = int(image_new.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(image_new, dim, interpolation = cv2.INTER_CUBIC) """
        top = top_part(top_resize)
        btm_resize = cv2.resize(image_new, None, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)
        btm = bottom_part(btm_resize)
        province = getProvince(btm)
        input_image_data_url = img_to_base64(image_new)
    else:
        img_resize = cv2.resize(img, None, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)
        top = top_part(img_resize)
        btm = bottom_part(img_resize)
        province = getProvince(btm)
        input_image_data_url = img_to_base64(image_new)

    return {'bbox': result,'detect_img': input_image_data_url, 'lp_number': top, 'province': province}

@app.post("/car-color/")
async def upload_image(request: Request, image: UploadFile = File()):
    contents = await image.read()

    # Read the uploaded image
    nparr = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    predict = model2(img)

    result = json.loads(predict.pandas().xyxy[0].to_json(orient="records"))

    x1 = int(result[0]['xmin'])
    y1 = int(result[0]['ymin'])
    x2 = int(result[0]['xmax'])
    y2 = int(result[0]['ymax'])

    image_new = img[y1: y2, x1: x2]

    resized = cv2.resize(image_new, None, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)
    
    color = detect_color(resized)

    input_image_data_url = img_to_base64(resized)

    return {'bbox': result,'detect_img': input_image_data_url, 'color': color}

def img_to_base64(img):
    img_buffer = cv2.imencode('.jpg', img)[1]
    img_base64 = base64.b64encode(img_buffer.tobytes()).decode('utf-8')

    return f'data:image/jpg;base64,{img_base64}'

