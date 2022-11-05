import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow.keras.utils import to_categorical
from .ImageProcessing import imageProcessing
from .FindCuttingLine import findCuttingLine
from .ProgressBar import progressBar

class dataLoading():
    '''
    INPUT : 
     -channel : 1,3
     -paths : list
     -classes : list, ex:[0,1,2]
     -rotation : list
     -handleCuttingLine: 切割道特徵萃取, default=False
     -deleteWithoutCuttingLine: 刪除錄無切割道的X,Y, default=False
    '''
    def __init__(self, channel, paths, classes=None, rotation=None, handleCuttingLine=False, deleteWithoutCuttingLine=False):
        self.channel=channel
        self.paths=paths
        if (classes is None)&(rotation is None)&(handleCuttingLine==False)&(deleteWithoutCuttingLine==False):
            self.output_X=self.gen_X()
        else :
            self.classes=classes if classes is not None else [0]*len(paths)
            self.rotation=rotation if rotation is not None else ['']*len(paths)
            self.unique_classes=list(set(classes)) if classes is not None else 1
            self.handleCuttingLine=handleCuttingLine
            self.deleteWithoutCuttingLine=deleteWithoutCuttingLine
            self.cuttingLine=[]
            self.cuttingLineSpec=[]

    def gen_Y(self):
        if len(self.unique_classes)>2:
            y=to_categorical(self.classes)
        else :
            y=self.classes
        return y
    
    def gen_X(self):
        paths_num=len(self.paths)
        skip_err=0
        X_list=[]
        for j in range(paths_num):
            try:
                image=cv2.imread(self.paths[j])
                if self.channel==1:
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                X_list.append((cv2.resize(image, (224, 224), interpolation = cv2.INTER_CUBIC))/255)
                
            except Exception as e:
                skip_err+=1
                print(self.paths[j],'-> skip! error')
                print(e)
                    
        X=np.array(X_list)
        if skip_err>0:
            print('skip_err=\t',skip_err)
        return X
        
    def gen_XY(self):
        '''
        1.先生成 y array (classes)
        2.影像處理
            2-1.若handleCuttingLine==True -> ROI影像處理
            2-2.若channel==1 -> 轉灰階
            2-2.其他 -> 轉RGB
        3.若deleteWithoutCuttingLine=True 且 profile is None -> 跳過不執行4
        4.若append image(224*224),classes to X_list,Y_list
        5.將X_list,Y_list轉np.array
        '''
        classes=self.gen_Y()
        skip_profile=0
        skip_err=0
        X_list=[]
        Y_list=[]
        fileName_list=[]
        starttime = time.time()
        paths_num=len(self.paths)
        print(f'gen_XY() is processing，共有 {paths_num} 筆資料')
        progress_Bar=progressBar()
        progress_Bar.start()
        cnt=0
        for j in range(paths_num):
            try:
                imgRotation=self.rotation[j] if self.rotation[j] is not None else ""
                image_origin=cv2.imread(self.paths[j])
                image_rotation=self.augmentation(image_origin,imgRotation)
                
                if self.handleCuttingLine==True:
                    '''此處image為dict，最多有兩張ROI後的圖片'''
                    profile,image=self.cutting_line_handler(image_rotation)
                else :
                    if self.channel==1:
                        image=cv2.cvtColor(image_rotation, cv2.COLOR_BGR2GRAY)
                        profile=None
                    else:
                        image=cv2.cvtColor(image_rotation, cv2.COLOR_BGR2RGB)
                        profile=None
                
                if (self.deleteWithoutCuttingLine==True) & (profile is None):
                    skip_profile+=1
                    print(self.paths[j],'-> skip! profile=None')
                else :
                    if type(image) is dict :
                        '''此處image若為dict --> values各別append to list '''
                        for key,item in image.items():
                            X_list.append((cv2.resize(item, (224, 224), interpolation = cv2.INTER_CUBIC))/255)
                            Y_list.append(classes[j])
                            fileName_list.append(os.path.split(self.paths[j])[-1])
                    else:
                        X_list.append((cv2.resize(image, (224, 224), interpolation = cv2.INTER_CUBIC))/255)
                        Y_list.append(classes[j])
                        fileName_list.append(os.path.split(self.paths[j])[-1])

            except Exception as e:
                skip_err+=1
                print(self.paths[j],'-> skip! error')
                print(e)
            
            finally:
                cnt+=1
                progress_Bar.update(cnt,paths_num)
                
        X=np.array(X_list)
        Y=np.array(Y_list)
        fileName=np.array(fileName_list)
        print('X_shape=\t',X.shape)
        print('Y_shape=\t',Y.shape)
        print('skip_profile=\t',skip_profile)
        print('skip_err=\t',skip_err)
        return X,Y,fileName
    
    def cutting_line_handler(self,image):
        prep=imageProcessing(image)
        binary_img=prep.hsv_segmentation((180, 255, 30))
        cutting_line=findCuttingLine(binary_img)
        profile=cutting_line.find_cutting_line_profile(0.2,1.5)
        if profile is not None:
            image=cutting_line.roi(prep.adaptive_Hist_equalize())
            result,final=self.checkProfileSpec(profile,0.3)
            self.cuttingLine.append(1)
            self.cuttingLineSpec.append(final)
        else :
            image=prep.gray
            self.cuttingLine.append(0)
            self.cuttingLineSpec.append(None)
        return profile, image
    
    def augmentation(self,image,imgRotation):
        if imgRotation=='hor':
            image=cv2.flip(image, 1)
        elif imgRotation=='ver':
            image=cv2.flip(image, 0)
        elif imgRotation=='hor_ver':
            image=cv2.flip(image, -1)
        elif imgRotation=='90':
            image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif imgRotation=='180':
            image=cv2.rotate(image, cv2.ROTATE_180)
        elif imgRotation=='270':
            image=cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
    
    def checkProfileSpec(self,profile,f):
        result={}
        judge={}
        for key,values in profile.items():
            if len(values)==7:
                try:
                    d1=values[2]-values[1]
                    d2=values[5]-values[4]
                    dmin=round((values[4]-values[2])*f)
                    result[key]=[d1,d2,dmin]
                    judge[key]=1 if (dmin<=d1)|(dmin<=d2) else 0
                except Exception as e:
                    print(e)
                    print(values)
            else :
                print(f'資料數錯誤= {len(values)} (==7)')
        final=1 if sum(judge.values())>=1 else 0
        return result,final
        