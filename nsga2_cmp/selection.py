from nsga2 import nondominance_sort,crowding_distance
import numpy as np

def select(pop,config):
    """
    evolution with select individual to stay

    :param pop: the trans_pop waiting to be selected
    :param config: the related parameter
    :return: 下一代种群
    """

    if config["model"]=="nsga2":
        pop_idx=select_nsga2(pop,config)
    elif config["model"]=="nsga2-cmp":
        pop_idx=select_nsga2_cmp(pop,config)
    else:
        raise Exception("Undefined model type")
    return pop_idx


def select_nsga2(pop,config):
    """
    only-nsga2

    :param pop: the trans_pop waiting to be selected
    :param config: the related parameter
    :return: the next generation
    """
    prob=config["problem"]()
    obj_value=prob.get_obj_values(pop)
    m=config["M"]
    d=config["d"]
    trans_n=pop.shape[0]
    n=config["pop_size"]
    sub_pop_count,dominateset=nondominance_sort.fast_nodominated_sorted(obj_value,trans_n)
    pop_res=np.empty((config["pop_size"],config["d"]))
    count = 0  # the count of have chosen individuals from population_new
    critical_level = -1  # the level of the critical population
    for i in sub_pop_count:
        count += sub_pop_count[i]
        if count > n:
            critical_level = i
            critical_num = sub_pop_count[i] # the number of the individual in the critical_level
            count-=sub_pop_count[i]
            break
        elif count == n:
            break
    # critical_level==-1: can completely choose the individual from a subpopulation
    if critical_level != -1:
        yet_choose_num = n - count  # the choosing number of critical population
        critical_obj = np.empty((critical_num, m))
        critical_ind = np.empty(
            (critical_num, 1))  # to track the index of a individual with the original population_new
        k = 0
        j = 0
        # choose the individuals in the  critical-level population
        for i in range(trans_n):
            if dominateset[i].rank < critical_level:
                pop_res[j, :] = pop[i, :]
                j += 1
            elif dominateset[i].rank == critical_level:
                critical_obj[k, :] = obj_value[i, :]
                critical_ind[k] = i
                k += 1
        # to calculate the crowding distance to maintain the diversity of the population
        # through the objective space
        pop_num = crowding_distance.crowding_distance_assignment(critical_obj, critical_num, prob.M, yet_choose_num)
        res_pop = np.empty((yet_choose_num, d))
        temp_row = 0
        for i in pop_num:
            res_pop[temp_row] = pop[int(critical_ind[i])]
            temp_row += 1
        temp_row = 0
        for i in range(j, n):
            pop_res[i] = res_pop[temp_row]
            temp_row += 1

    return pop_res


def select_nsga2_cmp(pop,config):
    pass

