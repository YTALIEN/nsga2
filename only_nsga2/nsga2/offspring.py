import numpy as np
import random

from config import mutation_pro,crossover_pro,problems


def sbx_crossover(individuals,problem,cros_pro=crossover_pro,eta_c=15):
    """
    simulated binary crossover
    reference from :CL-DDEA

    :param individuals:
    :param eta_c:
    :param problem: the string of test problem
    :return:
    """
    n,d=individuals.shape
    prob=problems[problem]()
    upp_bound=prob.get_upp_bound()
    low_bound=prob.get_low_bound()
    individuals_new=np.empty((0,d),individuals.dtype)
    offcro_size=0
    while offcro_size<n:
        # bug,这样设置似乎交叉概率没了意义
        if random.random()<cros_pro:
            rand_id1=np.random.choice(np.arange(0,n))
            rand_id2=np.random.choice(np.hstack((np.arange(0,rand_id1),np.arange(rand_id1,n))))

            p1_vec=np.minimum(individuals[rand_id1],individuals[rand_id2])
            p2_vec=np.maximum(individuals[rand_id1],individuals[rand_id2])

            sub=p2_vec-p1_vec
            beta_vec=1+2*np.minimum(p1_vec-low_bound,upp_bound-p2_vec)/sub
            alpha_vec=2-beta_vec**(-(eta_c+1))

            alpha_vec=np.where(np.isnan(alpha_vec),1,alpha_vec)

            u_vec=np.random.random(d)
            beta_q_vec = np.where(u_vec <= (1 / alpha_vec),
                                  (u_vec * alpha_vec) ** (1 / (eta_c + 1)),
                                  (1 / (2 - u_vec * alpha_vec)) ** (1 / (eta_c + 1))
                                  )
            # beta_q_vec = np.array(beta_q_vec, dtype=np.int)
            beta_q_vec = np.array(beta_q_vec)

            child1 = 0.5 * (p1_vec + p2_vec) - 0.5 * beta_q_vec * (p2_vec - p1_vec)
            child2 = 0.5 * (p1_vec + p2_vec) + 0.5 * beta_q_vec * (p2_vec - p1_vec)
            # set the bound
            child1 = child1.clip(low_bound, upp_bound)
            child2 = child2.clip(low_bound, upp_bound)
            # add the return population
            individuals_new = np.vstack((individuals_new, child1, child2))
            offcro_size=individuals_new.shape[0]

    return individuals_new

def pm_mutation(individuals_new, problem, mut_pro=mutation_pro, eta_m=15):
    """
    polynomial mutation

    :Reference
        code:cl-ddea
        information:https://blog.csdn.net/u013785405/article/details/86146903?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-86146903-blog-105446432.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-86146903-blog-105446432.pc_relevant_aa&utm_relevant_index=1

    :param individuals_new:
    :param problem:
    :param eta_m:
    :param mut_pro:

    :return

    """
    n,d=individuals_new.shape
    prob = problems[problem]()
    res = np.empty_like(individuals_new)
    for i in range(n):
        x_vec=individuals_new[i]
        ran = np.random.random(d)
        low_bound = prob.get_low_bound()
        upp_bound = prob.get_upp_bound()
        sub = upp_bound - low_bound
        temp = np.minimum(x_vec- low_bound, upp_bound - x_vec) / sub
        delta = np.where(ran <= 0.5,
                         (2 * ran + (1 - 2 * ran) * ((1 - temp) ** (eta_m + 1))) ** (1 / eta_m + 1) - 1,
                         1 - (2 * (1 - ran) + 2 * (ran - 0.5) * ((1 - temp) ** (eta_m + 1))) ** (1 / eta_m + 1)
                         )
        x_vec_new = x_vec+ delta * sub
        x_vec_new = x_vec_new.clip(low_bound, upp_bound)
        res[i] = np.where(np.random.random(d) < mut_pro, x_vec_new, x_vec)

    return res



