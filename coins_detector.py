# -----------------------------------------------------------
# Coins detector
#
# Author: Inés Prieto Centeno
# Put the camera parallel to a blank sheet of paper, adjust the lighting / height to be able to
# correctly detect the coins
# -----------------------------------------------------------

import cv2
from warp_perspective import Warp_perspective

def get_value_coins(area):
    """ get both the values of the coins and the coin that is given its area, returns 0 if none is found
    """
    if area > 3550 and area < 3750:
        print(area)
        return '2E', 2
    if area > 2950 and area < 3150:
        return '1E', 1
    if area > 2350 and area < 2550:
        return '5cnt', 0.05
    if area > 3200 and area < 3400:
        return '50cnt', 0.5
    if area > 2600 and area < 2800:
        return '20cnt', 0.2
    if area > 1400 and area < 1600:
        return '1cnt', 0.01
    if area > 2050 and area < 2250:
        return '10cnt', 0.1
    if area > 1800 and area < 2000:
        return '2cnt', 0.02
    return '0',0



video = cv2.VideoCapture(0)
url = "https://192.168.43.1:8080/video"
video.open(url) 
""" 
This was used to connect the webcam of the pc with that of the mobile
url = "https://mobilIP/video"
video.open(url) 
"""

while(True):
    _, image = video.read()

    # wp = Warp_perspective(frame=image, width=480, height=600)
    # image = wp.roi() 
    
    if image is not None:
        
        #image preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5),1)

        #find the positions of the coins by threshold and get the edge of the coins by findContours, then paint the edge with drawContours (-1 to paint)
        _,th2 = cv2.threshold(blur,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        cv2.imshow('coins_th',th2)
        cnts_2 = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image, cnts_2, -1, (255,0,0),2)

        total_sum = 0

        for c_2 in cnts_2:

            area = cv2.contourArea(c_2)
            momentos = cv2.moments(c_2)

            if momentos["m00"] == 0.0:
                momentos["m00"] = 1.0

            x = int(momentos["m10"]/momentos["m00"])
            y = int(momentos["m01"]/momentos["m00"])

            coinValueString, coinValue = get_value_coins(area) 
            if coinValueString != '0':
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, coinValueString , (x,y), font, 0.75, (0,255,0),2)
                total_sum += coinValue
        
        print("Total sum: "+str(total_sum))
        cv2.imshow("coins", image)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break