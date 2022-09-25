import sampling


def setting_params(kw):
    config = {
        "problem": None,
        "d": None,
        "M": None,
        "lower_bound": None,
        "upper_bound": None,
        "crossover_pro": 0.9,
        "mutation_pro": None,
        "eta_c": 15,
        "eta_m": 15,
        "max_gen": 100,
        "gen": None,
        "max_run": 20,
        "run": None,
        "pop_init": sampling.lhs,
        "pop_size": 100,
        "pop": None,
        "info_flag": True,
        "model": "nsga2",
    }

    config.update(kw)

    if config["lower_bound"] is None:
        config["lower_bound"] = config["problem"]().lower

    if config["upper_bound"] is None:
        config["upper_bound"] = config["problem"]().upper

    if config["d"] is None:
        config["d"] = config["problem"]().d

    if config["M"] is None:
        config["M"] = config["problem"]().M

    if (config["lower_bound"] is None) or (config["upper_bound"] is None):
        raise Exception("Undefined boundary")

    if config["mutation_pro"] is None:
        config["mutation_pro"] = 1 / config["d"]

    if config["pop"] is None:
        config["pop"] = config["pop_init"](
            n=config["pop_size"],
            d=config["d"],
            lower_bound=config["lower_bound"],
            upper_bound=config["upper_bound"]
        )

    return config
