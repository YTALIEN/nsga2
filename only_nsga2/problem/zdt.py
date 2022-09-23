import math
import numpy as np
from utils import seq


class problembase:
    def __init__(self):
        raise NotImplementedError

    def f1(self, individual):
        raise NotImplementedError

    def f2(self, individual):
        raise NotImplementedError

    def perfect_pareto_front(self):
        raise NotImplementedError


class ZDT1(problembase):
    def __init__(self):
        self.d = 30
        self.M = 2
        self.upper = np.ones(self.d)
        self.lower = np.zeros(self.d)

    def f1(self, individual):
        return individual[0]

    def f2(self, individual):
        sigma = sum(individual[1:])
        g = 1 + 9 * sigma / (self.d - 1)
        h = 1 - np.sqrt(self.f1(individual) / g)
        return g * h

    def perfect_pareto_front(self):
        '''

        :return: the x,y axis of PF
        '''
        domain = seq(0, 1, 0.01)
        return domain, list(map(lambda x1: 1 - math.sqrt(x1), domain))

    def get_low_bound(self):
        return self.lower

    def get_upp_bound(self):
        return self.upper

    def get_obj_values(self, individual):
        f1 = self.f1(individual)
        f2 = self.f2(individual)
        res = np.empty((1, 2))
        res[0][0] = f1
        res[0][1] = f2
        return res

    def get_objectivespace(self, individuals,n):
        """
            calculate the objective function
           :param individuals: the population of solutions
           :param n: population size
           :return: the population of the relative objective
           """
        obj_value = self.get_obj_values(individuals[0])
        for i in range(1, n):
            temp = self.get_obj_values(individuals[i])
            obj_value = np.vstack((obj_value, temp))
        return obj_value
