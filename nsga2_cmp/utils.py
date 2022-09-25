def seq(start, stop, step=1):
    """

    :param start: start of the dominance interval
    :param stop:  stop of the dominance interval
    :param step:
    :return: a sequence of [start,start+1*step,...,stop]
    """

    n = int(round((stop - start) / float(step)))
    if n > 1:
        return [start + step * i for i in range(n + 1)]
    else:
        return []


class RelationshipDominate:
    def __init__(self):
        self.dominated_count = 0
        self.dominate_set = set()
        self.rank = 0


def cmp(v1, v2):
    """
    the non/dominated relationship of two individuals
    :param v1: individual 1
    :param v2: individual 2
    :return:
    """
    if len(v1) != len(v2):
        return None
    flag = 0
    length = len(v1)
    for i in range(length):
        if v1[i] > v2[i]:
            flag = -1
            return flag
        elif v1[i] < v2[i]:
            flag = 1
        else:
            continue
    return flag
