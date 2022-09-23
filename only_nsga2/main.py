import random
import time
import numpy as np
import evolution
import config
import os
import matplotlib.pyplot as plt

file_path = 'test'

if not os.path.isdir(file_path):
    os.mkdir(file_path)


def main(problem=config.chosen_problem, pop_size=config.pop_size, mr=config.max_run):
    final_pop = []
    prob = config.problems[problem]()
    # randomly choose a run's result to show
    rand = random.randint(0, mr - 1)

    for i in range(mr):
        time1 = time.time()
        res = evolution.evolution_with_nsga2(problem, pop_size)
        time2 = time.time()
        print("\n{}:total_time={:.2f}s".format(i + 1, time2 - time1))
        final_pop.append(res)

        # show as a picture
        if i == rand:
            pareto_front = prob.get_objectivespace(res, pop_size)
            true_pf_x, true_pf_y = prob.perfect_pareto_front()
            plt.scatter(true_pf_x, true_pf_y, c='none', marker='o', edgecolors='g', label="true_pareto_front")
            plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='none', marker='o', edgecolors='r',
                        label="evaluate_pareto_front")
            plt.xlabel("f1")
            plt.ylabel("f2")
            plt.legend(loc='upper right')
            plt.show()

        config.run += 1
    np.savetxt(f'{file_path}/{problem}_result.csv', final_pop, delimiter="\t")


if __name__ == "__main__":
    main()
