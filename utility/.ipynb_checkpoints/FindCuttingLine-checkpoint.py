import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics

class findCuttingLine():
    def __init__(self, img):
        self.contours, self.hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.mask=np.zeros((img.shape[0],img.shape[1]))
        self.direction=None
        self.cutting_line_contour=None
        self.profile={}
        self.filter_size()
    
    def filter_size(self):
        if self.hierarchy is not None :
            max_cont=0
            for index in range(len(self.hierarchy[0])):
                cont=self.contours[index]
                x, y, w, h = cv2.boundingRect(cont)
                a = cv2.contourArea(cont)
                if a>13000 and w>40 and h>40 and (h>400 or w>400) and (a>max_cont):
                    max_cont=a
                    self.cutting_line_contour=cont

            if self.cutting_line_contour is not None :
                x, y0, w0, h0 = cv2.boundingRect(self.cutting_line_contour)
                self.direction='cross' if (w0/h0 > 0.7)&(w0/h0 < 1.5) else 'hor' if w0/h0 > 3 else 'ver'
    
    def find_cutting_line_profile(self,thres,roi_ratio):
        if self.cutting_line_contour is not None :
            #step1 fill contour
            filled_contour=self.fill_contour(self.cutting_line_contour,self.mask)
            #step2 extract profile 
            if self.direction=='hor':
                self.profile['hor']=self.profile_handler(filled_contour,'hor',thres,roi_ratio)
            elif self.direction=='ver':
                self.profile['ver']=self.profile_handler(filled_contour,'ver',thres,roi_ratio)
            elif self.direction=='cross':
                self.profile['hor']=self.profile_handler(filled_contour,'hor',thres,roi_ratio)
                self.profile['ver']=self.profile_handler(filled_contour,'ver',thres,roi_ratio)
            else:
                print('self.direction=None')
                
        else:
            return None
        return self.profile
    
    def profile_handler(self,array,direction,thres,roi_ratio):
        n=0 if direction=='ver' else 1
        
        #step1 stacking single axis counts
        axis_filled = np.array(array)[:,n]
        unique, counts = np.unique(axis_filled, return_counts=True)
        frequent_list = [unique[i] for i in range(len(counts)) if counts[i]==max(counts)]
        
        #step2 denoise for cross
        if self.direction=='cross':
            denoise_list=self.denoise(unique,counts,thres)
            denoised = [ x for x in axis_filled if x not in denoise_list]
        else :
            denoised=axis_filled
            
        d=int(abs((max(frequent_list)-min(frequent_list))*roi_ratio))
        roi_min=statistics.mean(frequent_list)-d if (statistics.mean(frequent_list)-d)>0 else 0
        roi_max=statistics.mean(frequent_list)+d
        return [roi_min, min(denoised), min(frequent_list), statistics.mean(frequent_list), max(frequent_list), max(denoised), roi_max]
    
    def fill_contour(self,contour,mask):
        filled_mask=cv2.drawContours(mask, [contour], -1, 1, -1)
        filled_contour=[]
        for x in range(filled_mask.shape[1]):
            for y in range(filled_mask.shape[0]):
                if filled_mask[y,x]==1:
                    filled_contour.append([x,y])
        return filled_contour
    
    def denoise(self,unique,counts,f):
        normalized_counts=[counts[i]/max(counts) for i in range(len(counts))]
        denoise_list=[unique[i] for i in range(len(normalized_counts)) if normalized_counts[i]<f]
        return denoise_list
        
    def find_cutting_line_contour(self):
        return self.cutting_line_contour if self.cutting_line_contour is not None else None
    
    def reversed_roi(self,image):
        if self.profile!={}:
            y1=self.profile['hor'][0] if 'hor' in self.profile.keys() else 10000
            y2=self.profile['hor'][6] if 'hor' in self.profile.keys() else 0
            x1=self.profile['ver'][0] if 'ver' in self.profile.keys() else 10000
            x2=self.profile['ver'][6] if 'ver' in self.profile.keys() else 0
            image[0:y1,0:x1],image[0:y1,x2:10000],image[y2:10000,0:x1],image[y2:10000,x2:10000]=255,255,255,255
        else :
            print("請先執行 : find_cutting_line_profile()")
        return image
    
    def roi(self,image):
        '''
        (1) 依照方向作ROI
        (2) hor方向轉90度
        (3) append to dict and finally return
        '''
        image_array={}
        if self.profile!={}:
            for key,item in self.profile.items():
                p1=item[0]
                p2=item[6]
                roi_image=image[p1:p2,0:10000] if key=='hor' else image[0:10000,p1:p2]
                roi_image_rotate=cv2.rotate(roi_image, cv2.ROTATE_90_CLOCKWISE) if key=='hor' else roi_image
                image_array[key]=roi_image_rotate
        return image_array
            
            
            




