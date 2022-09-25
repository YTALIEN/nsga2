from utils import seq
import numpy as np
import math


class ZDTProblemBase:
    def __init__(self):
        self.M = 2

    def f1(self, individual):
        return individual[0]

    def f2(self, individual):
        raise NotImplementedError

    def perfect_pareto_front(self):
        raise NotImplementedError

    def get_obj_values(self, individuals):
        n = individuals.shape[0]
        res = np.empty((n, self.M))
        for i in range(individuals.shape[0]):
            f1 = self.f1(individuals[i])
            f2 = self.f2(individuals[i])
            res[i][0] = f1
            res[i][1] = f2
        return res


class ZDT1(ZDTProblemBase):
    def __init__(self):
        super(ZDT1, self).__init__()
        self.name="ZDT1"
        self.d = 30
        self.upper = np.ones(self.d)
        self.lower = np.zeros(self.d)

    def f2(self, individual):
        sigma = sum(individual[1:])
        g = 1 + 9 * sigma / (self.d - 1)
        h = 1 - np.sqrt(self.f1(individual) / g)
        return g * h

    def perfect_pareto_front(self):
        """

        :return: the x,y axis of PF
        """
        domain = seq(0, 1, 0.01)
        return domain, list(map(lambda x1: 1 - math.sqrt(x1), domain))


class ZDT2(ZDTProblemBase):
    def __init__(self):
        super(ZDT2, self).__init__()
        self.name = "ZDT2"
        self.d = 30
        self.upper = np.ones(self.d)
        self.lower = np.zeros(self.d)

    def f2(self, individual):
        sigma = sum(individual[1:])
        g = 1 + 9 * sigma / (self.d - 1)
        h = 1 - ((self.f1(individual) / g) ** 2)
        return g * h

    def perfect_pareto_front(self):
        domain = seq(0, 1, 0.01)
        return domain, list(map(lambda x1: 1 - x1 ** 2, domain))


class ZDT3(ZDTProblemBase):
    def __init__(self):
        super(ZDT3, self).__init__()
        self.name = "ZDT3"
        self.d = 30
        self.upper = np.ones(self.d)
        self.lower = np.zeros(self.d)

    def f2(self, individual):
        sigma = sum(individual[1:])
        g = 1 + 9 * sigma / (self.d - 1)
        f1 = self.f1(individual)
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        return g * h

    def perfect_pareto_front(self):
        step = 0.01
        domain = seq(0, 0.0830015349, step) \
                 + seq(0.1822287280, 0.2577623634, step) \
                 + seq(0.4093136748, 0.4538821041, step) \
                 + seq(0.6183967944, 0.6525117038, step) \
                 + seq(0.8233317983, 0.8518328654, step)
        return domain, list(map(lambda x1: 1 - math.sqrt(x1) - x1 * math.sin(10 * math.pi * x1), domain))


class ZDT4(ZDTProblemBase):
    def __init__(self):
        super(ZDT4, self).__init__()
        self.name = "ZDT4"
        self.d = 10
        self.upper = np.hstack((1, 5 * np.ones(self.d - 1)))
        self.lower = np.hstack((0, -5 * np.ones(self.d - 1)))

    def f2(self, individual):
        sigma = sum(individual[1:] ** 2 - 10 * np.cos(4 * np.pi * individual[1:]))
        g = 1 + 10 * (self.d - 1) + sigma
        h = 1 - np.sqrt(self.f1(individual) / g)
        return g * h

    def perfect_pareto_front(self):
        domain = seq(0, 1, 0.01)
        return domain, list(map(lambda x1: 1 - math.sqrt(x1), domain))
