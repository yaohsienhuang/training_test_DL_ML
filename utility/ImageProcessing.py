import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics

class imageProcessing():
    def __init__(self, image):
        if isinstance(image, str):
            if os.path.exists(image):
                self.image=self.read_image()
            else:
                print("It's not a path.")
        else :
            self.image=image
        
    def read_image(self,path):
        image = cv2.imread(path)
        return image
        
    def gray(self):
        self.gray=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.gray
    
    def hsv(self):
        self.hsv=cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        return self.hsv
        
    def hsv_segmentation(self,hsv_region):
        self.hsv()
        black_region = cv2.inRange(self.hsv, (0, 0, 0), hsv_region)
        image_MORPH = cv2.morphologyEx(black_region, cv2.MORPH_CLOSE, np.ones((6,6),np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        image_dilated = cv2.dilate(image_MORPH, kernel, iterations=2)
        return image_dilated
    
    def adaptive_Hist_equalize(self):
        '''使用gray進行處理'''
        self.gray()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
        gray=self.gray.copy()
        cl1 = clahe.apply(gray)
        return cl1
    
    def binarization(self,image):
        threshold, image_bin = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        retval, bins = cv2.threshold(image_bin, threshold, 255, cv2.THRESH_BINARY)
        return retval
