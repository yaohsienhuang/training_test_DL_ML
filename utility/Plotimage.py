import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

class plotimage():
    def __init__(self, size, image_list):
        self.image_list=image_list
        self.n=len(image_list)
        plt.figure(figsize=(size,size*self.n))
        self.plot()
    
    def plot(self):
        for i in range(self.n):
            img=self.image_list[i]
            plt.subplot(1, self.n, i+1)
            if(len(img.shape)<3):
                plt.imshow(img, cmap='gray')
            else :
                plt.imshow(img)
        
def plot_hist(data, set_range, annotation=False):
    '''
    set_range = np.arange(0, 1.1, 0.02) / 20 / None 
    '''
    plt.figure(figsize=(15,6))
    values, bins, bars =plt.hist(data, density=False,bins=set_range)
    plt.xlabel('axis')
    plt.ylabel("Counts")
    if annotation==True:
        for bar in bars:
            plt.annotate(str(round(bar.get_height())), (bar.get_x(), bar.get_height()))
    plt.xticks(set_range)
    plt.show()
    
def plot_curve(xlabel,x,ylabel,y,annotation):
    plt.figure(figsize=(15,6))
    plt.plot(x,y)
    for i in range(len(x)):
        plt.annotate('%s'%annotation[i], xy=(x[i],y[i]))
    plt.grid()
    plt.suptitle(f'{xlabel}-{ylabel} Curve')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    