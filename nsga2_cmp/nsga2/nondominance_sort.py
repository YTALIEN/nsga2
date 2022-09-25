from utils import cmp, RelationshipDominate


def fast_nodominated_sorted(obj_values, pop_size):
    """
        to compare the pros and cons among the population

        :param obj_values: used for judge the non/dominated relationship
        :param pop_size: size of the population
        :return: sub_popdict: a dictionary including the number of individual in every level of the subpopulations
                dominateset： a list--every individual ,see in utils.RelationshipDominate
        """
    n = obj_values.shape[0]
    dominateset = []
    # initial
    for k in range(n):
        dominateset.append(RelationshipDominate())
    # compare and non/dominated between every two individuals
    for i in range(n):
        for j in range(n):
            is_dominate = cmp(obj_values[i, :], obj_values[j, :])
            if is_dominate == 1:
                # update the count and set of related individual
                dominateset[j].dominated_count += 1
                dominateset[i].dominate_set.add(j)
    count = 1  # the number of individual has been classified
    level = 1  # the index of the current subpopulation
    all_pop = set()
    for i in range(n):
        all_pop.add(i)
    havedone_pop = set()  # the set including individuals who was classified in a specific subpopulation

    while count <= pop_size:
        last_level_pop = set()
        rest_pop = all_pop - havedone_pop
        # choose the next subpopulation
        for i in rest_pop:
            if dominateset[i].dominated_count == 0:
                dominateset[i].rank = level
                havedone_pop.add(i)
                last_level_pop.add(i)
                count += 1
        level += 1
        # update the dominated inf
        for i in last_level_pop:
            for j in dominateset[i].dominate_set:
                dominateset[j].dominated_count -= 1

    sub_popdict = dict()
    for i in range(1, level):
        sub_popdict[i] = 0
    for i in range(n):
        sub_popdict[dominateset[i].rank] += 1
    return sub_popdict, dominateset
