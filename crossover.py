import random


def cxTwoPointCopy(ind1, ind2):
    """ Crossover to work with numpy array """
 
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2


def cxUniform(ind1, ind2):
    
    size = len(ind1) 

    for i in range(size):
        flip = random.randint(0, 1)
        if flip == 0: 
            ind1[i], ind2[i] = ind2[i], ind1[i] 
            
    return ind1, ind2 
