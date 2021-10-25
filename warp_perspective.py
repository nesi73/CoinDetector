import cv2
import numpy as np

class Warp_perspective:
    def __init__(self, frame, width, height):
        self.frame = frame
        self.width = width
        self.height = height

    def order_points(self, points):
        n_points = np.concatenate([points[0], points[1], points[2], points[3]]).tolist()
        y_order = sorted(n_points, key=lambda n_points : n_points[1])
        x1_order = y_order[:2]
        x1_order = sorted(x1_order, key=lambda x1_order : x1_order[0])
        x2_order = y_order[2:4]
        x1_order = sorted(x2_order, key=lambda x2_order : x2_order[0])

        return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

    def roi(self):
        aligned_image = None
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        _,th = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
        cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:1]

        for c in cnts:
            epsilon = 0.01*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)

            if len(approx) == 4:
                points = self.order_points(approx)
                pts1 = np.array(points, dtype='float32')
                pts2 = np.array([[0,0],[self.width,0],[0,self.height],[self.width,self.height]], dtype='float32')
                M = cv2.getPerspectiveTransform(pts1,pts2)
                aligned_image = cv2.warpPerspective(self.frame,M,(self.width,self.height))
        
        return aligned_image