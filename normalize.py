import numpy as np 
from read_data import read_data, save_data  
#from sklearn.preprocessing import normalize
import matplotlib.pyplot as plot 
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import cdist

def normalize(vector): 
    """ normalize vector, 
    sklearn.preprocessing did not work, don't know why 
    """ 
    min_val = min(vector)
    max_val = max(vector) 

    res = vector - min_val 
    res /= max_val - min_val

    return res
        


if __name__ == "__main__":

    template = read_data("template.txt").flatten()
    residuals = read_data("residuals.txt").flatten()
 
    norm_template = normalize(template)
    norm_residuals = normalize(residuals)

    print(min(template))
    print(max(template))

    print(min(norm_template))
    print(max(norm_template))


    f, (ax1, ax2) = plot.subplots(1, 2)
    ax1.imshow(template.reshape(26, 56), interpolation="none", cmap=plot.cm.gray)
    ax2.imshow(residuals.reshape(26, 56), interpolation="none",  cmap=plot.cm.gray)
    plot.show()

    print(pearsonr(norm_template, norm_residuals))
    print(cdist(np.atleast_2d(norm_template),
                np.atleast_2d(norm_residuals)))


#    save_data("template_nrm.txt", norm_template)

