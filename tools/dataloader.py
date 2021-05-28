import os
import numpy as np

class DataLoader:
    X_p = None
    X_n = None
    ratio = 0.85

    def __init__(self, ratio=0.85):
        if not os.path.exists('tools/data'):
            os.mkdir('tools/data')
        if not os.path.exists('tools/data/X_n.npy') or not os.path.exists('tools/data/X_p.npy'):
            raise Exception('Сannot load dataset')
        self.X_p = np.load('tools/data/X_p.npy', allow_pickle=True)
        self.X_n = np.load('tools/data/X_n.npy', allow_pickle=True)
        np.random.shuffle(self.X_p)
        np.random.shuffle(self.X_n)
        self.ratio = ratio
    
    def getTrainPositives(self):
        return self.X_p[0:int(len(self.X_p) * self.ratio)]
    
    def getTrainNegatives(self):
        return self.X_n[0:int(len(self.X_n) * self.ratio)]

    def getTestPositives(self):
        return self.X_p[int(len(self.X_p) * self.ratio):]
    
    def getTestNegatives(self):
        return self.X_n[int(len(self.X_n) * self.ratio):]
