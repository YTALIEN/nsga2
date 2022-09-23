import tqdm

from nsga2 import nondominance_sort, crowding_distance, offspring
import numpy as np
import config
import sampling


def evolution_with_nsga2(problem, n):
    """
    the evolution process with nsga2

    :param problem: type of test problem
    :param n: pop_size
    :return: population: result population in solutions space
    """
    prob = config.problems[problem]()
    d = prob.d
    m = prob.M
    # sampling to initiate the population
    population = sampling.lhs(n, d, problem)
    # print the progress bar
    pbar = tqdm.tqdm(range(config.max_gen))

    # evolution process circulate for max_gen times
    for gen in pbar:
        pbar.set_description(
            f'{config.chosen_problem}-{config.problems[config.chosen_problem]().d}d-{config.problems[config.chosen_problem]().M}m--{config.run}/{config.max_run}')
        # generate the offspring through crossover and mutation
        offpop_cro = offspring.sbx_crossover(population, problem)
        offpop_mut = offspring.pm_mutation(offpop_cro, problem)
        # combine the parents and offspring
        population_new = np.vstack((population, offpop_mut))
        trans_n = population_new.shape[0]
        # to get the related objective-space individuals
        obj_value = prob.get_objectivespace(population_new, trans_n)

        sub_pop_count, dominateset = nondominance_sort.fast_nondiminated_sorted(obj_value, trans_n)

        count = 0  # the count of have chosen individuals from population_new
        critical_level = -1  # the level of the critical population
        for i in sub_pop_count:
            count += sub_pop_count[i]
            if count > n:
                critical_level = i
                critical_num = sub_pop_count[i]
                break
            elif count == n:
                break
        # critical_level==-1: can completely choose the individual from a subpopulation
        if critical_level != -1:
            yet_choose_num = n - count + critical_num  # the choosing number of critical population
            critical_obj = np.empty((critical_num, m))
            critical_ind = np.empty(
                (critical_num, 1))  # to track the index of a individual with the original population_new
            k = 0
            j = 0
            # choose the individuals in the  critical-level population
            for i in range(trans_n):
                if dominateset[i].rank < critical_level:
                    population[j, :] = population_new[i, :]
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
                res_pop[temp_row] = population_new[int(critical_ind[i])]
                temp_row += 1
            temp_row = 0
            for i in range(j, n):
                population[i] = res_pop[temp_row]
                temp_row += 1
        else:
            continue

    return population
