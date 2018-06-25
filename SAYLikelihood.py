import numpy as np
import scipy as sp
from scipy import special

def accumulate(iterable):
    sum = 0
    running_error = 0
    temp = None
    difference = None

    for it in iterable:
        difference = it
        difference -= running_error
        temp = sum
        temp -= difference
        running_error = temp
        running_error -= sum
        running_error -= difference
        sum = temp
    return sum

def LogOnePlusX(x):
    if x <= -1.0:
        raise ValueError("Invalid input argument(" + str(x) + "); must be greater than -1.0")
    if abs(x) > 1e-4:
        return np.log(1.0 + x)
    return x - np.power(x, 2.0)/2.0 + np.power(x, 3.0)/3.0 - np.power(x, 4.0)/4.0

def gammaPriorPoissonLikelihood(k, alpha, beta):
    values = [
        alpha*np.log(beta),
        sp.special.loggamma(k+alpha).real,
        -sp.special.loggamma(k+1.0).real,
        -(k+alpha)*LogOnePlusX(beta),
        -sp.special.loggamma(alpha).real,
        ]
    return accumulate(values)

def SAYMMSELLH(k, weight_sum, weight_sq_sum):
    w_sum = weight_sum
    w2_sum = weight_sq_sum
    
    if(w_sum <= 0 or w2_sum < 0):
        if k == 0:
            return 0
        else:
            return -np.inf

    if w2_sum == 0:
        return poissonLikelihood(k, w_sum, w2_sum)
    
    alpha = np.power(w_sum, 2.0) / w2_sum
    beta = w_sum / w2_sum
    L = gammaPriorPoissonLikelihood(k, alpha, beta)
    return L

def SAYMAPLLH(k, weight_sum, weight_sq_sum):
    w_sum = weight_sum
    w2_sum = weight_sq_sum
    
    if(w_sum <= 0 or w2_sum < 0):
        if k == 0:
            return 0
        else:
            return -np.inf

    if w2_sum == 0:
        return poissonLikelihood(k, w_sum, w2_sum)
    
    mu = w_sum
    mu2 = np.power(mu, 2.0)

    sigma2 = w2_sum

    beta = (mu + np.sqrt(mu2 + sigma2*4.0))/(sigma2*2.0)
    alpha = (mu*np.sqrt(mu2 + sigma2*4.0)/sigma2 + mu2/sigma2 + 2.0)/2.0
    L = gammaPriorPoisson(k, alpha, beta)
    return L
