import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from .Plotimage import plot_curve

class estimatePerformance:
    def __init__(self,gt=None,pred=None):
        if gt is not None :
            self.computating(gt,pred)

    def computating(self,gt,pred):
        '''
        cross_tab[gt][pred]
        '''
        cross_tab=pd.crosstab(pred,gt,margins = True).reindex(columns=["OK",'NG','All'],index=["OK",'NG','All'], fill_value=0)
        print(cross_tab)
        self.total=cross_tab['All']['All']
        self.Act_P=cross_tab['NG']['All']
        self.Act_N=cross_tab['OK']['All']
        self.Pred_P=cross_tab['All']['NG']
        self.Pred_N=cross_tab['All']['OK']
        self.FP=cross_tab['NG']['OK']
        self.FN=cross_tab['OK']['NG']
        self.TN=cross_tab['OK']['OK']
        self.TP=cross_tab['NG']['NG']
        self.cross_tab=cross_tab
        
    def underkill(self):
        return round((self.FP/self.total)*100,2)

    def overkill(self):
        return round((self.FN/self.total)*100,2)

    def accuracy(self):
        return round(((self.TN+self.TP)/self.total)*100,2)
    
    def NG_recall(self):
        return round((self.TP/self.Act_P)*100,2)

    def NG_precision(self):
        return round((self.TP/self.Pred_P)*100,2)
    
    def OK_recall(self):
        return round((self.TN/self.Act_N)*100,2)

    def OK_precision(self):
        return round((self.TN/self.Pred_N)*100,2)
    
    def indicators(self,data_type=None,col_name='default'):
        '''example:
        performance=estimatePerformance(
            gt=dftest_data['class'],
            pred=dftest_data['pred_class']
        ).indicators(data_type='df')
        '''
        indicators_dic={
            'total':self.total,
            'OK':self.Act_N,
            'NG':self.Act_P,
            'NG_recall':self.NG_recall(),
            'NG_precision':self.NG_precision(),
            'OK_recall':self.OK_recall(),
            'OK_precision':self.OK_precision(),
            'underkill':self.FP,
            'underkill%':self.underkill(),
            'overkill':self.FN,
            'overkill%':self.overkill(),
            'accuracy':self.accuracy(),
        }
        
        if data_type=='df':
            res=pd.DataFrame.from_dict(indicators_dic, orient='index',columns=[col_name])
            return res
        else :
            return indicators_dic
    
    def confusion_matrix(self,gt,pred):
        labels = list(set(gt))
        cm = confusion_matrix(gt, pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
        disp.plot(xticks_rotation='vertical')
        plt.show()
        
    def thresholding(self,gt,threshold_list,score,thres_class=None,plot=False):
        '''example:
        threhold=[0.1,0.9,0.99,0.999]
        performance=estimatePerformance().thresholding(
            gt=dftest_data['class'],
            threshold_list=threhold,
            score=dftest_data['top1_score'],
            thres_class=dftest_data['pred_class'],
            plot_curve=True)
        '''
        print('threshold_list=\t',threshold_list)
        df=pd.DataFrame()
        for thres in threshold_list:
            print('--------------------------------')
            print('thres=\t',thres)
            if thres_class is None:
                pred=np.where(np.asarray(score)<thres,"OK","NG")
            else:
                pred=np.where((np.asarray(score)>thres)&(np.asarray(thres_class)=="OK"),"OK","NG")
            self.computating(gt,pred)
            indicators=self.indicators(data_type='df',col_name=thres)
            df=pd.concat([df,indicators],axis=1)
        
        if plot==True:
            plot_curve(
                xlabel='OK_recall',
                x=df.loc['OK_recall',:].values,
                ylabel='OK_precision',
                y=df.loc['OK_precision',:].values,
                annotation=threshold_list)
            
            plot_curve(
                xlabel='NG_recall',
                x=df.loc['NG_recall',:].values,
                ylabel='NG_precision',
                y=df.loc['NG_precision',:].values,
                annotation=threshold_list)
            
            plot_curve(
                xlabel='threshold',
                x=threshold_list,
                ylabel='accuracy',
                y=df.loc['accuracy',:].values,
                annotation=threshold_list)
        return df