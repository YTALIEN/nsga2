import numpy as np
import tqdm as tqdm

import offspring
import pre_proces
import selection


def main(**kw):
    config=pre_proces.setting_params(kw)

    pop=config["pop"]

    if config["info_flag"]:
        pbar=tqdm.tqdm(range(config["max_gen"]))
        pbar.set_description(
            f'{config["problem"].__name__}-{config["d"]}d-{config["M"]}M--{config["run"]}/{config["max_run"]}')
    else:
        pbar=range(config["max_gen"])
    for i in pbar:
        config["gen"]=i+1

        pop1=offspring.sbx_crossover(
            pop=pop,
            lower_bound=config["lower_bound"],
            upper_bound=config["upper_bound"],
            cp=config["crossover_pro"],
            eta_c=config["eta_c"]
        )

        pop2=offspring.pm_mutation(
            # pop=np.vstack((pop,pop1))
            pop=pop1,
            lower_bound=config["lower_bound"],
            upper_bound=config["upper_bound"],
            mp=config["mutation_pro"],
            eta_m=config["eta_m"]
        )

        pop=np.vstack((pop,pop2))

        pop=selection.select(
            pop=pop,
            config=config
        )

    return pop
