import threading

class predictionRunners():
    def __init__(self, jobs, model):
        self.jobs = jobs
        self.jobs_num = len(jobs)
        self.model = model
        self.threads = []
        self.prediction = {}
        
    def job_handler(self,job,i):
        pred = self.model.predict(job, batch_size=64)
        self.prediction[i]=pred
        
    def threading(self):
        for i in range(self.jobs_num):
            self.threads.append(threading.Thread(target=self.job_handler(self.jobs[i],i)))
        print("total threads = ",len(self.threads))
    
    def start(self):
        self.threading()
        
        for i in range(len(self.threads)):
            print(f'worker-{i} 執行中...')
            self.threads[i].start()
            
        for j in range(len(self.threads)):
            self.threads[j].join()
        print("completed threads = ",len(self.threads))
    
        pred=[]
        for i in range(self.jobs_num):
            pred.append(self.threads[i])
        return pred