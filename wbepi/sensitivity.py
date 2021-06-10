from SALib.sample import saltelli, latin
from SALib.analyze import sobol, fast, rbd_fast, delta, morris
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
import pandas as pd


def LHS(para, size):
    """
    Latin Hypercube Sampling Method.
    :param para={"num_vars": 2,
    "names": ["beta", "gamma",...],
    "bounds":[[0.1, 0.5], [0.1, 0.5],],
    "dists":["unif", "norm","lognorm",] ,  other distribution can be derived from unif
    }
    :param size=Sample size
    :return: numpy.array with size size*num_vars
    """
    lhs_sample = latin.sample(para, size)
    return lhs_sample


def PRCC(X, Y):
    """
    :param X: input,dim>2, numpy.array
    :param Y: output, dim=1, numpy.array
    :return: np.array
    """
    x = ot.Sample(X)
    a = np.zeros((len(Y), 1))
    a[:, 0] = Y
    y = ot.Sample(a)
    prcc_results = ot.CorrelationAnalysis_PRCC(x, y)
    return prcc_results


def Morris(para, X, Y):
    """
    Method of Morris
    more details can be see SALib
    :param para: {"num_vars": 2,
    "names": ["beta", "gamma",...],
    "bounds":[[0.1, 0.5], [0.1, 0.5],],
    "dists":["unif", "norm","lognorm",]
    :param X: sample, dim>2, numpy.array
    :param Y: modelling output, numpy.array, size=sample size
    :return: np.array,
    """
    prcc_results = morris.analyze(para, X, Y, conf_level=0.95, print_to_console=False, num_levels=4)
    return prcc_results["S1"]


def RBD_FAST(para, X, Y):
    """
    Random Balance Designs Fourier Amplitude Sensitivity Test
    more details can be see SALib
    :param para: {"num_vars": 2,
    "names": ["beta", "gamma",...],
    "bounds":[[0.1, 0.5], [0.1, 0.5],],
    "dists":["unif", "norm","lognorm",]
    :param X: sample, dim>2, numpy.array
    :param Y: modelling output, numpy.array, size=sample size
    :return: np.array, all positive
    """
    prcc_results = rbd_fast.analyze(para, X, Y, print_to_console=False)
    return prcc_results["S1"]
