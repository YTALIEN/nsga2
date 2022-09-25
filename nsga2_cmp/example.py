import random
import time
import matplotlib.pyplot as plt
from problem import zdt
import main
import os
import numpy as np

test_problem = ["ZDT1", "ZDT2", "ZDT3", "ZDT4"]
problem_list = {
    "ZDT1": zdt.ZDT1,
    "ZDT2": zdt.ZDT2,
    "ZDT3": zdt.ZDT3,
    "ZDT4": zdt.ZDT4,
}

file_path = {
    "file_path": "test",
    "fig_path": "image",
}

max_gen = 380
max_run = 3
info_flag = True
models = ["nsga2", "nsga2_cmp"]

for path in file_path:
    if not os.path.isdir(file_path[path]):
        os.mkdir(file_path[path])


def run_benchmark(pl, model=models[0], max_gen=max_gen, max_run=max_run, info_flag=info_flag):
    final_res = []
    rand = random.randint(0, max_run - 1)
    for i in range(max_run):
        # time1 = time.time()
        res = main.main(
            problem=pl,
            run=i + 1,
            max_gen=max_gen,
            max_run=max_run,
            info_flag=info_flag,
        )
        # time2 =time.time()
        # if info_flag:
        #     print(f'\n{i+1}:total_time ={time2-time1}\n')

        pareto_front = pl().get_obj_values(res)
        if i == rand:
        # if i == 1:
            true_pf_x, true_pf_y = pl().perfect_pareto_front()
            plt.scatter(true_pf_x, true_pf_y, c="green",label="true_pareto_front")
            plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='none', marker='o', edgecolors='r',
                        label="evaluate_pareto_front")
            plt.xlabel("f1")
            plt.ylabel("f2")
            plt.legend(loc='upper right')
            plt.title(f'{pl().name}-{max_gen}gens-PF')
            plt.savefig(f'{file_path["fig_path"]}/{pl().name}_{max_gen}gens_PF.jpg')
            plt.show()

        res_pf = pareto_front.reshape(-1)
        final_res.append(res_pf)
    final_res = np.array(final_res)
    np.savetxt(f'{file_path["file_path"]}/{pl().name}_{model}.csv', final_res, delimiter="\t")


if __name__ == "__main__":
    for prob in test_problem[3:]:
        # for model in models:
        # run_benchmark(problem_list[prob],model)
        run_benchmark(problem_list[prob])
