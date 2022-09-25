import numpy as np


def lhs(n, d, lower_bound, upper_bound):
    sub = upper_bound - lower_bound
    if np.any(sub < 0):
        return None
    interval_size = 1.0 / n
    sample_points = np.empty([n, d])
    for i in range(n):
        sample_points[i, :] = np.random.uniform(low=i * interval_size, high=(i + 1) * interval_size, size=d)
    sample_points = lower_bound + sample_points * sub
    for i in range(d):
        np.random.shuffle(sample_points[:, i])
    return sample_points


def rs(n, d, lower_bound, upper_bound):
    sub = upper_bound - lower_bound
    if np.any(sub < 0):
        return None
    sample_points = np.random.random([n, d])
    sample_points = lower_bound + sample_points * sub
    return sample_points
