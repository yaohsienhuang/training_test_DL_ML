import numpy as np
import gc
from .ProgressBar import progressBar
from .DataLoading import dataLoading

class predictBatch():
    '''
    examplle:
       pred=predictBatch(
            model=model,
            paths=dftest_data['fullPath'].values,
            channel=1,
            n=5000,
            batch_size=128).prediction
    '''
    def __init__(self,model,paths,channel,n,batch_size,rounded=True):
        self.model=model
        self.n=n
        self.batch_size=batch_size
        self.prediction=self.paths_to_predict(channel,paths,rounded)

    def paths_to_predict(self,channel,paths,rounded):
        '''
        totol= 3324 -> n= 1000 -> split= 4 -> 831 pcs/batch
        '''
        num=len(paths)
        k0, m = divmod(num, self.n)
        k=k0 if m==0 else k0+1
        paths_array=np.array_split(paths,k)
        prediction_list=[]
        progress_Bar=progressBar()
        progress_Bar.start()
        cnt=0
        for i in range(len(paths_array)):
            path_i=paths_array[i]
            testX=dataLoading(channel=channel, paths=path_i).output_X
            prediction = self.model.predict(testX, batch_size=self.batch_size)
            for pred in prediction:
                if rounded==True:
                    prediction_list.append([round(num, 5) for num in pred.tolist()])
                else:
                    prediction_list.append(pred)
            cnt+=len(path_i)
            progress_Bar.update(cnt,num)
            
            testX,prediction=None,None
            del testX
            gc.collect()
        return prediction_list