import numpy as np
import matplotlib.pyplot as plot 
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import cdist

def read_data(name): 
    
    vec = open(name,"r").read().split() 
    vec = np.array(list(map(float, vec)))
    
    picture = vec.reshape(56, 26)
    picture = picture.transpose() 

    return picture     

def save_data(name, matrix):
    
    with open(name, "w") as f:
        print(matrix.shape)
        matrix = matrix.reshape(26, 56)
        matrix = matrix.transpose()
        print(matrix.shape)
        for line in matrix:
            for x in line:
                print(x, file=f)

def show(matrix):

    plot.imshow(matrix, interpolation="none", cmap=plot.cm.gray)
    plot.show()

    
if __name__ == "__main__":

    mouth = read_data("mouth.txt")
    show(mouth)

    template = read_data("template.txt") 
    show(template) 

    res = read_data("residuals.txt")
    show(res)

    # res2 = mouth-template
    # show(res2) 

    print(pearsonr(template.flatten(), res.flatten()))
    print(cdist(np.atleast_2d(template.flatten()), 
                np.atleast_2d(res.flatten())))
