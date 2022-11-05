import os
import cv2
import numpy as np
import pandas as pd
import shutil
import math
import random
import xml.etree.ElementTree as ET
from .ProgressBar import ProgressBar

class fileProcessing():
    def __init__(self, path=None):
        self.image_extension = ["jpg","JPG","jpeg","JPEG","webp","WEBP","bmp","BMP","png","PNG"]
        if path!=None:
            self.path=os.path.normpath(path)+os.sep
            self.dir_list=self.get_dir_list(self.path)
            self.image_list=self.get_file_list(self.path)
    
    def merge_dirs(self,sources,target,mode):
        '''sample:
        fileProcessing().merge_dirs(
                    sources=['1','2','3'],
                    target='4',
                    mode='copy'
        )
        '''
        target=os.path.normpath(target)+os.sep
        for sour in sources:
            self.path=os.path.normpath(sour)+os.sep
            self.image_list=self.get_file_list(self.path)
            self.file_transfer_with_frac(f=1,target=target,mode=mode)
    
    def file_transfer_by_df(self,df,column_list,target):
        '''sample:
        fileProcessing().file_transfer_by_df(
                    df=dfpath,
                    column_list=['fullPath','customer','folder','defect'],
                    target='backup/test/'
        )
        '''
        # convert data from df to dict
        col_num=len(column_list)
        data_dict={}
        item_num=0
        for i in range(col_num):
            col_name=column_list[i]
            data_dict[col_name]=df[col_name].values
            item_num=len(df[col_name].values)
        print('item_num=\t',item_num)
        # restructuring and iterating
        progress_Bar=progressBar()
        progress_Bar.start()
        cnt=0
        for j in range(item_num):
            target_path=os.path.normpath(target)
            for key,item in data_dict.items():
                if 'path' in key.lower():
                    source_path=data_dict[key][j]
                else:
                    subdir=data_dict[key][j]
                    target_path=target_path+os.sep+subdir
            if not os.path.isdir(target_path+os.sep):
                os.makedirs(target_path+os.sep,exist_ok=True)
            shutil.copy(source_path,target_path+os.sep)
            cnt+=1
            progress_Bar.update(cnt,item_num)
    
    def get_file_list(self,path,extension=None):
        if extension is None:
            extension=self.image_extension
        file_list = []
        print(f'extension=\t{extension}')
        print(f'start to collect paths from {path}')
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                fullPath = os.path.join(maindir, filename)
                ext = fullPath.split('.')[-1]
                if ext in extension:
                    file_list.append(fullPath)
        print(f'total path=\t{len(file_list)}')
        return file_list
    
    def get_dir_list(self,path):
        dir_list=os.listdir(path)
        print(f'this folder has {len(dir_list)} dirs')
        return dir_list
    
    def add_extension(self,add_list):
        self.image_extension.extend(add_list)
        print(f'extension=\t{self.image_extension}')
    
    def file_transfer_with_frac(self,f,target,mode,replace_dict=None):
        '''example:
        file_transfer=fileProcessing(path='/tf/cp1ai01/COG/03_POC訓練資料/label/FM_model/NG/FM')
        file_transfer.file_transfer_with_n(
            f=0.1,
            target='/tf/cp1ai01/COG/03_POC訓練資料/label/FM_model/NG/FM_00',
            mode='move'
        )
        '''
        target=os.path.normpath(target)+os.sep
        num=len(self.image_list)
        choiced_num=int(round(num*f,0))
        choice_list=np.random.choice(self.image_list, choiced_num, replace=False)
        print(f'ramdom smaple=\t{len(choice_list)}/{num}')
        print(f'start to {mode} files to {target}')
        progress_Bar=progressBar()
        progress_Bar.start()
        cnt=0
        for path in choice_list:
            subdirs=path.replace(self.path,'')
            if replace_dict!=None:
                for word, replacement in replace_dict.items():
                    subdirs = subdirs.replace(word, replacement)
            target_path=target+os.path.split(subdirs)[0]+os.sep
            if not os.path.isdir(target_path):
                os.makedirs(target_path,exist_ok=True)
            if mode=='move':
                shutil.move(path,target_path)
            if mode=='copy':
                shutil.copy(path,target_path)
            cnt+=1
            progress_Bar.update(cnt,choiced_num)
        
    def file_transfer_with_n(self,n,target,mode,replace_dict=None):
        '''example:
        file_transfer=fileProcessing(path='/tf/cp1ai01/COG/03_POC訓練資料/label/FM_model/NG/FM')
        file_transfer.file_transfer_with_n(
            n=500,
            target='/tf/cp1ai01/COG/03_POC訓練資料/label/FM_model/NG/FM_00',
            mode='move'
        )
        '''
        target=os.path.normpath(target)+os.sep
        num=len(self.image_list)
        choiced_num=int(round(n,0))
        choice_list=np.random.choice(self.image_list, choiced_num, replace=False)
        print(f'ramdom smaple=\t{len(choice_list)}/{num}')
        print(f'start to {mode} files to {target}')
        progress_Bar=progressBar()
        progress_Bar.start()
        cnt=0
        for path in choice_list:
            subdirs=path.replace(self.path,'')
            if replace_dict!=None:
                for word, replacement in replace_dict.items():
                    subdirs = subdirs.replace(word, replacement)
            target_path=target+os.path.split(subdirs)[0]+os.sep
            if not os.path.isdir(target_path):
                os.makedirs(target_path,exist_ok=True)
            if mode=='move':
                shutil.move(path,target_path)
            elif mode=='copy':
                shutil.copy(path,target_path)
            cnt+=1
            progress_Bar.update(cnt,choiced_num)
        
    def split_folder_with_n(self,n,target,mode,start_num=0):
        '''sample:
        file_processor=fileProcessing(path='/tf/cp1ai01/COG/03_POC訓練資料/backup/split_test/test')
        file_processor.split_folder_with_n(
            n=500,
            target='/tf/cp1ai01/COG/03_POC訓練資料/object_detection/FM_model-preparing',
            mode='move',
            start_num=1
        )
        '''
        target=os.path.normpath(target)+os.sep
        total_n=len(self.image_list)
        folder_num=math.ceil(total_n/n)
        print(f'Each folders contains {n} pcs -> {total_n} pcs should split to {folder_num} folders.')
        random.shuffle(self.image_list)
        folder_cnt=start_num
        progress_Bar=progressBar()
        progress_Bar.start()
        cnt=0
        for i in range(total_n):
            path=self.image_list[i]
            subdir=os.path.split(path)[0].split(os.sep)[-1]
            target_path=target+f'{subdir}-{folder_cnt}'+os.sep
            if not os.path.isdir(target_path):
                os.makedirs(target_path,exist_ok=True)
            if mode=='move':
                shutil.move(path,target_path)
            elif mode=='copy':
                shutil.copy(path,target_path)
            cnt+=1
            progress_Bar.update(cnt,choiced_num)
                
    def extract_both_exist_image_xml(self,path,target,mode,image_extension='JPG'):
        '''sample:
        fileProcessing().extract_both_exist_image_xml(
            path='/tf/cp1ai01/COG/03_POC訓練資料/object_detection/FM_model-preparing/FM_OK-0',
            target='/tf/cp1ai01/COG/03_POC訓練資料/object_detection/FM_model/FM_OK',
            mode='copy',
        )
        '''
        target=os.path.normpath(target)+os.sep
        xml_list=self.get_file_list(path,['xml'])
        total_n=len(xml_list)*2
        print('xml+image=\t',total_n)
        progress_Bar=progressBar()
        progress_Bar.start()
        cnt=0
        for path in xml_list:
            if not os.path.isdir(target):
                os.makedirs(target,exist_ok=True)
            image_path=path.replace('.xml', f'.{image_extension}')
            if mode=='move':
                shutil.move(path,target)
                shutil.move(image_path,target)
            elif mode=='copy':
                shutil.copy(path,target)
                shutil.copy(image_path,target)
            cnt+=2
            progress_Bar.update(cnt,total_n)
            
    def read_xml_label_counts(self,path):
        xml_list=self.get_file_list(path,['xml'])
        result=[]
        for xml in xml_list:
            tree = ET.parse(xml)
            root = tree.getroot()
            for elem in root:
                if elem.tag=='object':
                    name = elem.find('name').text
                    result.append(name)
        values, counts = np.unique(result, return_counts=True)
        print('------ the count of labels ------')
        for i in range(len(values)):
            print(f'{values[i]}=\t{counts[i]} labels')
            
        
            
        


