from config import pop_size, problems
import numpy as np


def lhs(n, d, problem):
    """Latin hypercube sampling

    :param n: the number of sample data
    :param d: the dimension of decision vector
    :param problem: the type of test problem
    :return: a matrix of n*d
    """
    prob = problems[problem]()
    upp_bound = prob.get_upp_bound()
    low_bound = prob.get_low_bound()
    sub = upp_bound - low_bound
    if np.any(sub < 0):
        return None
    intervalSize = 1.0 / n
    samplePoints = np.empty([n, d])
    for i in range(n):
        samplePoints[i, :] = np.random.uniform(low=i * intervalSize, high=(i + 1) * intervalSize, size=d)
    samplePoints = low_bound + samplePoints * sub
    for i in range(d):
        np.random.shuffle(samplePoints[:, i])
    return samplePoints


def rs(n, d, problem):
    """Random sampling

    :param n: the number of sample data
    :param d: the dimension of decision vector
    :param problem: the type of test problem
    :return:a matrix of n*d
    """
    prob = problems[problem]()
    upp_bound = prob.get_upp_bound()
    low_bound = prob.get_low_bound()
    sub = upp_bound - low_bound
    if np.any(sub < 0):
        return None
    samplePoints = np.random.random([n, d])
    samplePoints = low_bound + samplePoints * sub
    return samplePoints
