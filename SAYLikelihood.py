import numpy as np
import scipy as sp
from scipy import special
from scipy import stats

def gammaPriorPoissonLikelihood(k, alpha, beta):
    """Poisson distribution marginalized over mean priored
       with a gamma distribution that has parameters (alpha, beta)

    Parameters
    ----------
    k : int
        The number of observed events
    alpha : float
        Gamma distribution shape parameter
    beta : float
        Gamma distribution rate parameter
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
    """Computes Log of the Poisson Likelihood

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

def SAYMMSELLH(k, weight_sum, weight_sq_sum):
    """Computes Log of the SAY Likelihood using Minimum Mean Squared Estimator (MMSE)

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

def SAYMAPLLH(k, weight_sum, weight_sq_sum):
    """Computes Log of the SAY Likelihood using Maximum A'postiori Probability (MAP) estimator

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
    L = gammaPriorPoisson(k, alpha, beta)
    return L
