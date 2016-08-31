import sys 
import random
import signal 
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import cdist

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import matplotlib.pyplot as plot

from fitness import Fitness
from crossover import cxTwoPointCopy, cxUniform

from read_data import read_data 

# uncomment for parallel run 
#from scoop import futures

NGEN = 1000000
CXPB = 0.6
MUTPB = 0.1
IND_LEN = 1456 
index = 1
NAME = "correlation_adversary"

# hall of fame
hof = None    
target = read_data("template_nrm.txt").flatten()

def handler(signum, frame):
    global target_image 
    global hof
    
   
    # show current solution 
    print(min(hof[0]), max(hof[0]))

    corr = pearsonr(hof[0], target)
    print(corr)
    dist = cdist(np.atleast_2d(hof[0]), np.atleast_2d(target))
    print(dist)

    f, (ax1, ax2) = plot.subplots(1, 2)

    ax1.imshow(target.reshape(26, 56), interpolation="none", cmap=plot.cm.gray)
    ax2.imshow(hof[0].reshape(26, 56), interpolation="none", cmap=plot.cm.gray)
    plot.show()

    # set the handler again, plot ate it
    signal.signal(signal.SIGINT, handler)

    x = input("Exit? y/n")
    if x == "y":
        sys.exit() 
    

def mainGA():
    """ Runs the main loop of GA.""" 
    global toolbox 
    global hof 

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(2, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
  
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, 
                                   ngen=NGEN, stats=stats, halloffame=hof, 
                                   verbose=True)

    return hof[0] 


# weights = (1.0,) stands for one objective fitness
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


if __name__ == "__main__": 

    signal.signal(signal.SIGINT, handler)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_real", random.random    )
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_real, IND_LEN)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) 
#    toolbox.register("map", futures.map)

    X = [] 
    # Run the GA for each target image and target output.
    for target_image in range(1):
        print("Target image: {} ".format(target_image))
        sys.stdout.flush()
        fit = Fitness()

        #Genetic operators 
        toolbox.register("evaluate", fit.evaluate)
        #toolbox.register("mate", cxTwoPointCopy) 
        toolbox.register("mate", cxUniform) 
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.01, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        #toolbox.register("select", tools.selRoulette) 

        X.append(mainGA())
            
 
    # save X to file 
    np.save(NAME, X)
 


