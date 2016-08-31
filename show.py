import numpy as np 
import matplotlib.pyplot as plot 
from keras.models import model_from_json 
from keras.datasets import mnist 
from matplotlib import rcParams
from scipy.stats.stats import pearsonr
from read_data import read_data 


FILE_NAME = "correlation_adversary.npy"

# Load matrix of adversary inputs.
# one image per line 
X = np.load(FILE_NAME) 
print(X.shape)

# Load mnist data set. 
#(X_train, y_train), (X_test, y_test) = mnist.load_data() 
#X_train = X_train.reshape(60000, 784)
#X_train = X_train.astype('float32') 
#X_train /= 255 

target = read_data("mouth.txt").flatten() 
print(target.shape)

#rcParams.update({'font.size': 8})


for i in range(1):
    
    corr = pearsonr(X[i], target)
    print(corr)

    # plot adversary image
    f, (ax1, ax2) = plot.subplots(1, 2)

    x = X[i].reshape(26, 56)
    ax1.imshow(x, interpolation="none", cmap=plot.cm.gray)
    ax2.imshow(target.reshape(26, 56), interpolation="none", cmap=plot.cm.gray)
    plot.show()

#plot.show()

    

