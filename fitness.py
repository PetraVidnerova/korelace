from keras.datasets import mnist 
from scipy.spatial.distance import cdist
from scipy.stats.stats import pearsonr
import numpy as np
import random
from read_data import read_data


class Fitness:

    def __init__(self):
        # Target image 
        self.target = read_data("template_nrm.txt").flatten()  
        


    def evaluate(self, individual):

        dist = cdist(np.atleast_2d(individual), np.atleast_2d(self.target))
        #dist /= len(self.target)
        
        corr = pearsonr(individual, self.target)[0] 

        # pearsonr returns couple (coef, p-value)
        #dist2 = np.abs(pearsonr(individual, self.target)[0])
            
        #coef = 0.0005 
        #fit = dist*(1-coef) + coef*dist2

        if dist < 5.0:
            dist = 5.0 

        fit = dist + abs(corr)

        return fit,  
        
