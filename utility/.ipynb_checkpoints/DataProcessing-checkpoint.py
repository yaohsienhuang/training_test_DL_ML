import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics

class dataProcessing():
    def __init__(self, paths=[] ,split_symbol='-'):
        '''sample:
            data_processor=dataProcessing(paths=['/tf/cp1ai01/COG/03_POC訓練資料/DM/COG FV/'], split_symbol='-')
            cols={
                'fullPath':'fullPath',
                'path-5':'customer',
                'path-6':'folder',
                'path-7':'class',
                'path-8':'defect',
                'fileName-0':'recipe',
                'fileName-1':'lotid',
                'fileName-2':'waferid',
                'fileName-3':'die'
            }
            dftest_data=test_processor.setup_columns(col_dict=cols, class_col='defect',class_col_replace={'PD':'FM'})
            dftest_data
        '''
        self.image_extension = ["jpg","JPG","jpeg","JPEG","webp","WEBP","bmp","BMP","png","PNG"]
        if len(paths)>0:
            self.paths=paths
            self.split_symbol=split_symbol
            self.classes_dic={}
            self.classes_dic_reverse={}
            self.df=None
            self.gen_df()
    
    def gen_df(self):
        data=[]
        cols=[]
        cnt=0
        for path in self.paths:
            path=os.path.normpath(path)
            for dirPath,dirNames,fileNames in os.walk(path):
                for fileName in fileNames:
                    fullPath=os.path.join(dirPath,fileName)
                    temp_data=[]
                    temp_data.extend([fullPath,fileName])
                    if fileName.split('.')[-1] in self.image_extension:
                        
                        # append dirname split
                        path_base=dirPath.split(os.sep)
                        temp_data.extend(path_base)

                        # append filename split
                        fileName=fileName.replace(' - 複製','').replace(' - copy','').replace(' - Copy','')
                        fileName_base=fileName.split(self.split_symbol)
                        if fileName_base[0]=='A':
                            fileName_base.pop(0)
                        temp_data.extend(fileName_base)
                        
                        # append cols
                        if cnt==0:
                            cols.extend(['fullPath','fileName'])
                            for i in range(len(path_base)):
                                cols.append(f'path-{i}')
                            for i in range(len(fileName_base)):
                                cols.append(f'fileName-{i}')
                    
                        data.append((temp_data))
                        cnt+=1
            print('path=\t',path)
            print('已完成(累積)=\t',len(data))
            
        length_diff=True
        for j in data:
            if len(j)!=len(cols):
                print(f'{len(j)}!={len(cols)} -> {j}')
                length_diff=False
                
        if length_diff==True:
            df=pd.DataFrame(data,columns=cols)
            self.df=df
            print(df.head(10))
    
    def setup_columns(self, col_dict, class_col, class_col_replace={}):
        '''
        col_dict={"path-0":'fileNmae',"path-1":'fullPath'}
        '''
        self.class_col=class_col
        col_list = self.df.columns.values.tolist()
        for col in col_list:
            if col not in col_dict.keys():
                self.df.pop(col)
        
        self.df=self.df.rename(columns=col_dict)
        self.df=self.df.replace({self.class_col:class_col_replace})
        self.gen_classes_dict(self.df)
        self.plot_hist(self.df,'total')
        return self.df
    
    def gen_classes_dict(self,df):
        classes_list=df[self.class_col].unique().tolist()
        special_classes = ['OK','ok','FM_OK','Chipping_OK','Overkill','overkill']
        temp_classes = set(classes_list).difference(special_classes)
        classes_list = [a for a in special_classes if a in classes_list] + sorted(list(temp_classes))
        classes_num=0
        for cla in classes_list:
            self.classes_dic[cla]=classes_num
            self.classes_dic_reverse[classes_num]=cla
            classes_num+=1
        print('classes_dic=\t',self.classes_dic)
        print('classes_dic_reverse=\t',self.classes_dic_reverse)
        
    def split_train_valid(self,df,f,seed,dirpath):
        dftrain=df.sample(frac=f,random_state=seed)
        dfvalid=df.drop(dftrain.index)
        print('shape(dftrain)=\t',dftrain.shape)
        print('shape(dfvalid)=\t',dfvalid.shape)
        self.plot_hist(dftrain,'train')
        self.plot_hist(dfvalid,'valid')
        
        print('trainset.xlsx 儲存中...')
        trainset=self.df.set_index(["recipe",self.class_col]).groupby(level=["recipe",self.class_col]).size().to_frame('counts').reset_index()
        with pd.ExcelWriter(dirpath + 'trainset.xlsx') as writer:
            trainset.to_excel(writer, sheet_name='TRNSET',index=False)
            dftrain.to_excel(writer, sheet_name='dfPath_train',index=False)
            dfvalid.to_excel(writer, sheet_name='dfPath_val',index=False)
            writer.save()
            writer.close()
        print(f'{dirpath}trainset.xlsx 已儲存。')
        
        return dftrain,dfvalid
    
    def plot_hist(self,df,title):
        fig, ax = plt.subplots()
        print(f'{title}=\n',df[self.class_col].value_counts())
        ax=df[self.class_col].value_counts().plot.bar()
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()))
        plt.title(title)
        plt.show()
        
    def pivot_table(self,df,values,index,columns,aggfunc):
        '''example:
        dataProcessing().pivot_table(
            df=dfpath,
            values='fullPath',
            index=['defect'],
            columns=['class'],
            aggfunc='count'
        )
        '''
        table = pd.pivot_table(
               data=df,
               values=values, 
               index=index,
               columns=columns,
               aggfunc=aggfunc
        )
        return table