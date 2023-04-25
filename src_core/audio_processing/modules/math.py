import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

def itp(x, px, py, axis=-1):
    return CubicSpline(px, py, axis=axis)(x)

def itpmono(x, px, py, axis=-1):
    return PchipInterpolator(px, py, axis=axis)(x)

def ols0v(y):
    """
    0-variate OLS

    Parameters
    y: 1d array of shape (nsamples,)

    Returns
    cons: float. The intercept of fitted line.
    res: float. The sum of squared residuals.
    """
    (cons,), (res,) = np.linalg.lstsq(np.ones((y.size, 1)), y, rcond=None)[:2]
    return cons, res

def ols1v(y, x):
    """
    1-variate OLS

    Parameters
    y: 1d array of shape (nsamples,)
    x: 1d array of shape (nsamples,)

    Returns
    cons: float. The intercept of fitted line.
    coef: float. The coefficient of x.
    res: float. The sum of squared residuals.
    """
    coef, (res,) = np.linalg.lstsq(np.vstack([np.ones(x.size), x]).T, \
                                   y, rcond=None)[:2]
    return coef[0], coef[1], res
    
def olsmv(y, X):
    """
    multi-variate OLS
    
    Parameters
    y: 1d array of shape (nsamples,)
    X: 2d array of shape (nsamples, nx)

    Returns
    cons: float. The intercept of fitted line.
    coef: 1d array of shape (nx,). The coefficients of X.
    res: float. The sum of squared residuals.
    """
    coef, (res,) = np.linalg.lstsq(np.append(np.ones((y.size, 1)), X, axis=1), \
                                   y, rcond=None)[:2]
    return coef[0], coef[1:], res
