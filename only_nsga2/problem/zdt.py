import math
import numpy as np
from utils import seq

""" 
reference:《Comparison of Multiobjective Evolutionary Algorithms: Empirical Results》

"""


class problembase:
    def __init__(self):
        self.d = None
        self.M = None
        self.upper = None
        self.lower = None
        raise NotImplementedError

    def f1(self, individual):
        raise NotImplementedError

    def f2(self, individual):
        raise NotImplementedError

    def perfect_pareto_front(self):
        raise NotImplementedError

    def get_obj_values(self,individual):
        f1 = self.f1(individual)
        f2 = self.f2(individual)
        res = np.empty((1, 2))
        res[0][0] = f1
        res[0][1] = f2
        return res

    def get_objectivespace(self, individuals, n):
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
        return super().get_obj_values(individual)

    def get_objectivespace(self, individuals, n):
        return super().get_objectivespace(individuals, n)


class ZDT2(problembase):
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
        h = 1 - (self.f1(individual) / g) ** 2
        return g * h

    def perfect_pareto_front(self):
        domain = seq(0, 1, 0.01)
        return domain, list(map(lambda x1: 1 - x1 ** 2, domain))

    def get_obj_values(self, individual):
        return super().get_obj_values(individual)

    def get_objectivespace(self, individuals, n):
        return super().get_objectivespace(individuals, n)


class ZDT3(problembase):
    def __init__(self):
        self.d = 30
        self.M = 2
        self.upper = np.ones(self.d)
        self.lower = np.zeros(self.d)

    def f1(self, individual):
        return individual[0]

    def f2(self, individual):
        sigma=sum(individual[1:])
        g = 1 + 9 * sigma / (self.d - 1)
        f1=self.f1(individual)
        h=1-np.sqrt(f1 / g)-(f1/g)*np.sin(10*np.pi*f1)
        return g*h

    def perfect_pareto_front(self):
        step = 0.01
        domain = seq(0, 0.0830015349, step) \
                 + seq(0.1822287280, 0.2577623634, step) \
                 + seq(0.4093136748, 0.4538821041, step) \
                 + seq(0.6183967944, 0.6525117038, step) \
                 + seq(0.8233317983, 0.8518328654, step)
        return domain, list(map(lambda x1: 1 - math.sqrt(x1) - x1 * math.sin(10 * math.pi * x1), domain))

    def get_obj_values(self, individual):
        return super().get_obj_values(individual)

    def get_objectivespace(self, individuals, n):
        super().get_objectivespace(individuals, n)

class ZDT4(problembase):

    def __init__(self):
        self.d = 10
        self.M = 2
        self.upper = np.hstack((1,5*np.ones(self.d-1)))
        self.lower = np.hstack((0,-5*np.ones(self.d-1)))

    def f1(self, individual):
        return individual[0]

    def f2(self, individual):
        sigma=sum(individual[1:]**2-10*np.cos(4*np.pi*individual[1:]))
        g=1+10*(self.d-1)+sigma
        h=1-np.sqrt(self.f1(individual)/g)
        return g*h

    def perfect_pareto_front(self):
        domain = seq(0, 1, 0.01)
        return domain, list(map(lambda x1: 1 - math.sqrt(x1), domain))

    def get_obj_values(self, individual):
        return super().get_obj_values(individual)

    def get_objectivespace(self, individuals, n):
        return super().get_objectivespace(individuals, n)


