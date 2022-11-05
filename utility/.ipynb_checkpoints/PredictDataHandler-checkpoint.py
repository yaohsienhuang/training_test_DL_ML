import os
import cv2
import numpy as np
import pandas as pd

class predictDataHandler():
    def __init__(self, prediction):
        self.prediction=prediction
        print(f'output dimension= {len(prediction[0])}')
        self.prediction_list=[]
        self.score_list=[]
        
    def binay_classes(self,threshold,export_df=False):
        self.top='top1'
        self.prediction_list=np.where(np.array(self.prediction)[:,0]>threshold,"NG","OK").tolist()
        self.score_list=np.array(self.prediction)[:,0].tolist()
        if export_df==True:
            return self.convert_df()
        else :
            return self.prediction_list, self.score_list
    
    def multi_classes(self,replace_dict=None,top='top1',export_df=False):
        '''example:
        pred_handler=predictDataHandler(pred)
        df=pred_handler.multi_classes(top='top1',replace_dict=classes_dic_reverse,export_df=True)
        '''
        self.top=top
        if top=='top3':
            for pred in self.prediction:
                cla=np.argsort(pred,axis=0)[::-1][:3].tolist()
                score=np.array(pred)[cla].tolist()
                cla_trans=[replace_dict[x] for x in cla] if replace_dict != None else cla
                self.prediction_list.append(cla_trans)
                self.score_list.append(score)
        else :
            cla=np.array(self.prediction).argmax(axis=-1)
            cla_trans=[replace_dict[x] for x in cla] if replace_dict != None else cla
            self.prediction_list=cla_trans
            self.score_list=np.amax(self.prediction,axis=1)
            
        if export_df==True:
            return self.convert_df()
        else :
            return self.prediction_list, self.score_list
    
    def convert_df(self):
        if self.top=='top3':
            top_dict={
                'top1_class':np.array(self.prediction_list)[:,0],
                'top1_score':np.array(self.score_list)[:,0],
                'top2_class':np.array(self.prediction_list)[:,1],
                'top2_score':np.array(self.score_list)[:,1],
                'top3_class':np.array(self.prediction_list)[:,2],
                'top3_score':np.array(self.score_list)[:,2]
            }
            df=pd.DataFrame(top_dict)
        else:
            df=pd.DataFrame({'top1_class':self.prediction_list,'top1_score':self.score_list})
        print(f'convert to dataframe with shape={df.shape}')
        return df