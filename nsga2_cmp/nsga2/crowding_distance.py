import numpy as np


def crowding_distance_assignment(obj_values, n, m, cn):
    """
    to calculate the crowding distance to choose individual from critical population
    :param obj_values: matrix of objective values
    :param n: the size of population
    :param m: the number of objectives
    :param cn: the rest number of the next generation
    :return: the index of chosen individual of the critical population
    """
    pop_distance = np.zeros((n, 1))
    index = np.arange(0, n)
    index = index.reshape(n, 1)
    obj_values = np.hstack((index, obj_values))
    for i in range(1, m + 1):
        # sorted by i-th objectives
        temp = obj_values[obj_values[:, i].argsort()]
        for j in range(1, n - 1):
            id_i = int(temp[j][0])
            # the objetive value of  i+i and i-1
            obj_value_ip1 = temp[j + 1][i]
            obj_value_is1 = temp[j - 1][i]
            pop_distance[id_i] = pop_distance[id_i] + (obj_value_ip1 - obj_value_is1)
        # boundary point
        pop_distance[int(temp[0][0])] = float("inf")
        pop_distance[int(temp[n - 1][0])] = float("inf")
    pop_distance = np.hstack((index, pop_distance))
    pop_distance[:, 1] = -pop_distance[:, 1]
    sorted_des = pop_distance[pop_distance[:, 1].argsort()]
    res = list()
    for i in range(cn):
        res.append(int(sorted_des[i][0]))
    return res
