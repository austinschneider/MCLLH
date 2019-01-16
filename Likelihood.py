import numpy as np
import scipy as sp
from scipy import special
from scipy import stats

def gammaPriorPoissonLikelihood(k, alpha, beta):
    """Poisson distribution marginalized over the rate parameter, priored with
       a gamma distribution that has shape parameter alpha and inverse rate
       parameter beta.

    Parameters
    ----------
    k : int
        The number of observed events
    alpha : float
        Gamma distribution shape parameter
    beta : float
        Gamma distribution inverse rate parameter

    Returns
    -------
    float
        The log likelihood
    """
    values = [
            alpha*np.log(beta),
            sp.special.loggamma(k+alpha).real,
            -sp.special.loggamma(k+1.0).real,
            -(k+alpha)*np.log1p(beta),
            -sp.special.loggamma(alpha).real,
            ]
    return np.sum(values)

def poissonLikelihood(k, weight_sum, weight_sq_sum):
    """Computes Log of the Poisson Likelihood.

    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum : float
        the sum of the weighted MC event counts
    weight_sq_sum : float
        the sum of the square of the weighted MC event counts

    Returns
    -------
    float
        The log likelihood
    """

    return sp.stats.poisson.logpmf(k, weight_sum)

def LMean(k, weight_sum, weight_sq_sum):
    """Computes Log of the L_Mean Likelihood.
       This is the poisson likelihood with gamma distribution prior where the
       mean and variance are fixed to that of the weight distribution.

    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum : float
        the sum of the weighted MC event counts
    weight_sq_sum : float
        the sum of the square of the weighted MC event counts

    Returns
    -------
    float
        The log likelihood
    """

    # Return -inf for ill formed an likelihood or 0 without observation
    if weight_sum <= 0 or weight_sq_sum < 0:
        if k == 0:
            return 0
        else:
            return -np.inf

    # Return the poisson likelihood in the appropriate limiting case
    if weight_sq_sum == 0:
        return poissonLikelihood(k, weight_sum, weight_sq_sum)

    alpha = np.power(weight_sum, 2.0) / weight_sq_sum
    beta = weight_sum / weight_sq_sum
    L = gammaPriorPoissonLikelihood(k, alpha, beta)
    return L

def LMode(k, weight_sum, weight_sq_sum):
    """Computes Log of the L_Mode Likelihood.
       This is the poisson likelihood with gamma distribution prior where the
       mode and variance are fixed to that of the weight distribution.

    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum : float
        the sum of the weighted MC event counts
    weight_sq_sum : float
        the sum of the square of the weighted MC event counts

    Returns
    -------
    float
        The log likelihood
    """

    # Return -inf for ill formed an likelihood or 0 without observation
    if weight_sum <= 0 or weight_sq_sum < 0:
        if k == 0:
            return 0
        else:
            return -np.inf

    # Return the poisson likelihood in the appropriate limiting case
    if weight_sq_sum == 0:
        return poissonLikelihood(k, weight_sum, weight_sq_sum)

    mu = weight_sum
    mu2 = np.power(mu, 2.0)

    sigma2 = weight_sq_sum

    beta = (mu + np.sqrt(mu2 + sigma2*4.0))/(sigma2*2.0)
    alpha = (mu*np.sqrt(mu2 + sigma2*4.0)/sigma2 + mu2/sigma2 + 2.0)/2.0
    L = gammaPriorPoissonLikelihood(k, alpha, beta)
    return L

def LEff(k, weight_sum, weight_sq_sum):
    """Computes Log of the L_Eff Likelihood.
       This is the poisson likelihood, using a poisson distribution with
       rescaled rate parameter to describe the Monte Carlo expectation, and
       assuming a uniform prior on the rate parameter of the Monte Carlo.
       This is the main result of the paper arXiv:1901.04645

    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum : float
        the sum of the weighted MC event counts
    weight_sq_sum : float
        the sum of the square of the weighted MC event counts

    Returns
    -------
    float
        The log likelihood
    """

    # Return -inf for ill formed an likelihood or 0 without observation
    if weight_sum <= 0 or weight_sq_sum < 0:
        if k == 0:
            return 0
        else:
            return -np.inf

    # Return the poisson likelihood in the appropriate limiting case
    if weight_sq_sum == 0:
        return poissonLikelihood(k, weight_sum, weight_sq_sum)

    alpha = np.power(weight_sum, 2.0) / weight_sq_sum
    beta = weight_sum / weight_sq_sum
    L = gammaPriorPoissonLikelihood(k, alpha, beta)
    return L

def computeLMean(k, weights):
    """Computes Log of the L_Mean Likelihood from a list of weights.
       This is the poisson likelihood with gamma distribution prior where the
       mean and variance are fixed to that of the weight distribution.

    Parameters
    ----------
    k : int
        the number of observed events
    weights : [float]
        the list of the weighted MC events

    Returns
    -------
    float
        The log likelihood
    """
    w = np.asarray(weights)
    weight_sum = np.sum(w)
    weight_sq_sum = np.sum(w*w)
    return LMean(k, weight_sum, weight_sq_sum)

def computeLMode(k, weights):
    """Computes Log of the L_Mode Likelihood from a list of weights.
       This is the poisson likelihood with gamma distribution prior where the
       mode and variance are fixed to that of the weight distribution.

    Parameters
    ----------
    k : int
        the number of observed events
    weights : [float]
        the list of the weighted MC events

    Returns
    -------
    float
        The log likelihood
    """
    w = np.asarray(weights)
    weight_sum = np.sum(w)
    weight_sq_sum = np.sum(w*w)
    return LMode(k, weight_sum, weight_sq_sum)

def computeLEff(k, weights):
    """Computes Log of the L_Eff Likelihood from a list of weights.
       This is the poisson likelihood, using a poisson distribution with
       rescaled rate parameter to describe the Monte Carlo expectation, and
       assuming a uniform prior on the rate parameter of the Monte Carlo.
       This is the main result of the paper arXiv:1901.04645

    Parameters
    ----------
    k : int
        the number of observed events
    weights : [float]
        the list of the weighted MC events

    Returns
    -------
    float
        The log likelihood
    """
    w = np.asarray(weights)
    weight_sum = np.sum(w)
    weight_sq_sum = np.sum(w*w)
    return LEff(k, weight_sum, weight_sq_sum)

