# -----------------------------------------------------------
# Coins detector
#
# Author: Inés Prieto Centeno
# Put the camera parallel to a blank sheet of paper, adjust the lighting / height to be able to
# correctly detect the coins
# -----------------------------------------------------------

import cv2
from warp_perspective import Warp_perspective
import os
import numpy as np

def is_a_coin(area):
    return area > 1400 and area < 3950

def normalize_labels(elements, image):
    a, b = [], []
    i = 0
    for elem in elements:
        a.append(elem) if i%2 == 0 else b.append(elem)
        i += 1

    x_min, x_max, y_min, y_max = min(a), max(a), min(b), max(b)
    width = (x_max - x_min)
    heigth = (y_max - y_min)
    image_width = np.shape(image)[1]
    image_heigth = np.shape(image)[0]

    x_center = (width/2 + x_min)
    y_center = (heigth/2 + y_min)

    return x_center / image_width, y_center / image_heigth, width / image_width, heigth / image_heigth

def write_file(file, label_normalize):
    f=open(file,"w+")
    f.write(str(0) + label_normalize)
    f.close()

name_folder = 'train'
folder = os.listdir("detected_coins/" + name_folder + "/images/")
cont = 0
for file in folder:
    image = cv2.imread('detected_coins/' + name_folder + '/images/' +file)
    resize_image = cv2.resize(image, (640,480), interpolation=cv2.INTER_AREA)
    
    if resize_image is not None:
        
        #image preprocessing
        gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5),1)

        #find the positions of the coins by threshold and get the edge of the coins by findContours, then paint the edge with drawContours (-1 to paint)
        _,th2 = cv2.threshold(blur,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        cnts_2 = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        
        label_normalize = ' '
        for c_2 in cnts_2:
            
            area = cv2.contourArea(c_2)            
            momentos = cv2.moments(c_2)

            if is_a_coin(area):
                x_center_normalize, y_center_normalize, width_normalize, heigth_normalize = normalize_labels(c_2.flatten(), resize_image)

                label_normalize += str(x_center_normalize) + ' ' + str(y_center_normalize) + ' ' + str(width_normalize) + ' ' + str(heigth_normalize)

        cont += 1
        file = 'detected_coins/' + name_folder + '/labels/0000' + str(cont) + '.txt'
        write_file(file, label_normalize)