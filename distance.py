import numpy as np 
import random 
from normalize import normalize

from scipy.stats.stats import pearsonr
from scipy.spatial.distance import cdist

from read_data import read_data 

template = read_data("template.txt").flatten() 
residuals = read_data("residuals.txt").flatten() 

template = normalize(template)
residuals = normalize(residuals)

dist = cdist(np.atleast_2d(template), np.atleast_2d(residuals)) 
corr = pearsonr(template, residuals) 

print(" template - residuals ") 
print("Distance: ", dist)
print("Correlation: ", corr)
print()

random_image = [ random.random() for i in range(len(template)) ]
dist = cdist(np.atleast_2d(template), np.atleast_2d(random_image)) 
corr = pearsonr(template, random_image) 

print(" template - random ") 
print("Distance: ", dist)
print("Correlation: ", corr)
print()

