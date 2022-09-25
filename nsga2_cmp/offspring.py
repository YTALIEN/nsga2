import random
import numpy as np

import warnings

# for the code: beta_vec = 1 + 2 * np.minimum(p1_vec - lower_bound, upper_bound - p2_vec) / (p2_vec - p1_vec)
# ignore message: divided by zero
warnings.filterwarnings("ignore")


def sbx_crossover(pop, lower_bound, upper_bound, cp=0.9, eta_c=15):
    """
        simulated binary crossover
        reference from :CL-DDEA

        :param pop: population
        :param eta_c: the division index
        :param lower_bound:
        :param upper_bound:
        :return: new_pop after crossover
    """
    n, d = pop.shape
    pop_new = np.empty((0, d), pop.dtype)
    cros_num = 0  # 控制交叉之后产生的子代的个数
    while cros_num < n:
        rand_p1 = np.random.choice(np.arange(0, n))
        rand_p2 = np.random.choice(np.hstack((np.arange(0, rand_p1), np.arange(rand_p1, n))))

        p1_vec = np.minimum(pop[rand_p1], pop[rand_p2])
        p2_vec = np.maximum(pop[rand_p1], pop[rand_p2])

        sub = p2_vec - p1_vec
        beta_vec = 1 + 2 * np.minimum(p1_vec - lower_bound, upper_bound - p2_vec) / sub
        alpha_vec = 2 - beta_vec ** (-(eta_c + 1))

        alpha_vec = np.where(np.isnan(alpha_vec), 1, alpha_vec)

        u_vec = np.random.random(d)
        beta_q_vec = np.where(u_vec <= (1 / alpha_vec),
                              (u_vec * alpha_vec) ** (1 / (eta_c + 1)),
                              (1 / (2 - u_vec * alpha_vec)) ** (1 / (eta_c + 1))
                              )
        # beta_q_vec = np.array(beta_q_vec, dtype=np.int)
        beta_q_vec = np.array(beta_q_vec)

        child1 = 0.5 * (p1_vec + p2_vec) - 0.5 * beta_q_vec * (p2_vec - p1_vec)
        child2 = 0.5 * (p1_vec + p2_vec) + 0.5 * beta_q_vec * (p2_vec - p1_vec)
        # set the bound
        child1 = child1.clip(lower_bound, upper_bound)
        child2 = child2.clip(lower_bound, upper_bound)

        if random.random() <= cp:
            pop_new = np.vstack((pop_new, child1, child2))
        else:
            pop_new = np.vstack((pop_new, pop[rand_p1], pop[rand_p2]))
        cros_num = pop_new.shape[0]
    return pop_new


def pm_mutation(pop, lower_bound, upper_bound, mp, eta_m=15):
    """
        polynomial mutation

        :Reference
            code:cl-ddea
            information:https://blog.csdn.net/u013785405/article/details/86146903?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-86146903-blog-105446432.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-86146903-blog-105446432.pc_relevant_aa&utm_relevant_index=1

        :param pop: population
        :param eta_m:
        :param mp: mutation_probability

        :return: new population after mutation

        """
    n, d = pop.shape
    pop_res = np.empty_like(pop)
    for i in range(n):
        x_vec = pop[i]
        ran = np.random.random(d)
        sub = upper_bound - lower_bound
        temp = np.minimum(x_vec - lower_bound, upper_bound - x_vec) / sub
        delta = np.where(ran <= 0.5,
                         (2 * ran + (1 - 2 * ran) * ((1 - temp) ** (eta_m + 1))) ** (1 / eta_m + 1) - 1,
                         1 - (2 * (1 - ran) + 2 * (ran - 0.5) * ((1 - temp) ** (eta_m + 1))) ** (1 / eta_m + 1)
                         )
        x_vec_new = x_vec + delta * sub
        x_vec_new = x_vec_new.clip(lower_bound, upper_bound)
        pop_res[i] = np.where(np.random.random(d) < mp, x_vec_new, x_vec)

    return pop_res
