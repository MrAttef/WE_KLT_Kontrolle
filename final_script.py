# -*- coding: utf-8 -*-
"""
Created on Mon Jun  12 22:14:09 2025

@author: Atef Kh
"""
import glob
import os
import time
import string
import cx_Oracle
import json
import serial
import re
import copy
import torch 
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from pypylon import pylon
from transformers import AutoProcessor, AutoModelForCausalLM
from opcua import Client
from ultralytics import YOLO
from torchvision import models, transforms


barcode=''
trigger= 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_loading = time.time()
yolo_model_path = './Yolo_model.pt' #pfad anpassen
defect_model_path = './Defect_model.pt' #pfad anpassen

model_defect = YOLO(defect_model_path) 
model_yolov9 = YOLO(yolo_model_path)
model_ResNet = torch.load(
    r".\resnet18_best_model_alpha75_gamma3.pt",
    map_location=device,
    weights_only=False 
).to(device).eval()
end_loading = time.time()
print("Model loaded in :", end_loading - start_loading, "seconds")

# Bild-Transformation
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

#Scanner function
def scan_barcode():
        # local variables 
    global barcode, trigger
    print("\nScanning KLT barcode \n")
    # Define serial port vCLEARariables
    port='COM4'
    baudrate=115200
    parity = serial.PARITY_EVEN
    stopbits = serial.STOPBITS_ONE
    bytesize = serial.EIGHTBITS
    timeout = 0.5
    trigger = 1
    # initialize scanner 
    scanner = serial.Serial(port, baudrate, bytesize, parity, stopbits, timeout)
    scanner.write(b'LON\r') # Trigger SR scanner with LON. \r is character return
    barcode = scanner.readline().decode()
    if (barcode==""): # if the barcode could not be read during the set time out, stop scanning and return error
        trigger=0
        barcode="Error"
        #scanner.write(b'LOFF\r')
    elif (len(barcode) != 11):
        trigger=0
        print("\nBarcode not read correctly! Trying again\n")
    print(barcode)   
    scanner.close()
    return barcode, trigger

#Camera functions
def init_camera():
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("no camera found")
    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
    camera.Open()
    return camera

def capture_image(camera):

    img = pylon.PylonImage()
    camera.ExposureTime.Value =   2000
    camera.StartGrabbing()
    with camera.RetrieveResult(1000) as result:
        img.AttachGrabResultBuffer(result)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        #filename1 = f"./klts/Image_{timestamp}.png"
        filename = f"/Users/e1519249/Downloads/BA/codes/klts/final_test/main.png"
        img.Save(pylon.ImageFileFormat_Png, filename)
        #img.Save(pylon.ImageFileFormat_Png, filename1)

        img.Release()
    camera.StopGrabbing()
    return img, timestamp

#Model functions

def load_images():
    image_files = glob.glob("/Users/e1519249/Downloads/BA/codes/klts/image/Objekte/*.png")  # pfad anpassen
    return image_files

def annotate(img, text):
    img_np = np.array(img)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale= 1
    color     = (0, 255, 0) if text == "RICHTIG" else (0, 0, 255)
    thickness = 2
    # Text oben links
    cv2.putText(img_np, text,
                org=(10, 30),
                fontFace=font,
                fontScale=font_scale,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA)
    
    return Image.fromarray(img_np)

def main():
    camera = init_camera()   # Uncomment um camera zu initialisieren

    while True:
        trigger = 0  # auf 0 setzen, wenn Barcode gelesen wird
        x = 0
        Objects_folder_path = './image/Objekte'  # Pfad anpassen
        os.makedirs(Objects_folder_path, exist_ok=True)
        barcode1, trigger = scan_barcode()  # Uncomment
        if trigger == 1:
            Start = time.time()
            img, timestamp=capture_image(camera)  # Uncomment um bilder aufzunehmen
            for filename in os.listdir(Objects_folder_path):
                file_path = os.path.join(Objects_folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            path = "./final_test/main.png" # Pfad anpassen


            Start = time.time()
            loadImage = Image.open(path).convert("RGB")
            input_tensor = transform(loadImage).unsqueeze(0).to(device)  
            with torch.no_grad():
                outputs = model_ResNet(input_tensor)
                _, pred = torch.max(outputs, 1)

            class_names = ['Falsch', 'Richtig']  

            results = model_yolov9(loadImage)  
            for idx, result in enumerate(results):
                xyxy = result.boxes.xyxy.cpu().numpy()  
                names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
                image = loadImage
                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    cropped = image.crop((x1, y1, x2, y2))
                    class_name = names[i] if i < len(names) else f"class_{i}"
                    #save_path = os.path.join(Objects_folder_path, f"{class_name}_{idx}_{i}_{timestamp}.png")
                    save_path = os.path.join(Objects_folder_path, f"{class_name}_{idx}_{i}.png")
                    if class_name == "irrelevent" or class_name == "empty":
                        x = 1
                    if class_name == "box":
                        if model_defect(cropped) == "Defects" or model_defect(cropped) == "Defect":
                            x = 1
                    cropped.save(save_path)

            if (class_names[pred.item()] == "Richtig") & (x == 0) :
                print("Vorhersage für die KLT Belegung: Richtig")
                ende = time.time()
                duration = ende - Start
                label = "RICHTIG " + barcode1 + "Dauer" + str(duration)[:5]
                out_img = annotate(loadImage, label)
                label = "RICHTIG"
                filename2  = f"./final_test/Img_{timestamp}_{duration:.3f}_{label}.jpg" # speicher Pfad anpassen
                out_img.save(filename2)
            else:
                print("Vorhersage für die KLT Belegung: Falsch")
                ende = time.time()
                duration = ende - Start
                label = "FALSCH " + barcode1 +"Dauer" + str(duration)[:5]
                out_img = annotate(loadImage, label)
                label = "FALSCH" 
                filename2  = f"./final_test/Img_{timestamp}_{duration:.3f}_{label}.jpg" # speicher Pfad anpassen
                out_img.save(filename2)


            ende = time.time()
            print("die Verarbeitungzeit des Bildes ist", ende - Start)
            time.sleep(10)



if __name__ == '__main__':
    main()
    print("Prozess beendet")