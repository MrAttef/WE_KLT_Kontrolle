# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:14:09 2025

@author: Atef Kh
"""
from pypylon import pylon
import time
import string
import cx_Oracle
import json
import serial
import re
import os
from opcua import Client


barcode=''
trigger= 0

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
    timeout = 1
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

def init_camera():
    # Verwende den ersten verfügbaren Pylon-Kameratyp
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("no camera found")
    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
    camera.Open()
    return camera
 
def capture_image(camera,barcode):
    img = pylon.PylonImage()
    #print(camera.GevSCPSPacketSize)
    #camera.AutoPacketSize = True
    camera.ExposureTime.Value =   1500
    camera.StartGrabbing()
    with camera.RetrieveResult(1500) as result:  # Timeout in Ms
        img.AttachGrabResultBuffer(result)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"./klts/Image_{timestamp}.png"
        img.Save(pylon.ImageFileFormat_Png, filename)
        img.Release()
    camera.StopGrabbing()
    return img

def main():
    camera = init_camera()
    while True:
        try:
            start=time.time()
            trigger = 0
            barcode,trigger = scan_barcode()
            if trigger == 1:
                img = capture_image(camera,barcode)
                ende=time.time()
                print("Time taken to capture image:", ende-start)
                time.sleep(15) # Timeout count how long the load carrier take to pass the camera
        except Exception as e :
           print("restarting ...")
           time.sleep(2)
           continue


main()
print("Programm beendet")