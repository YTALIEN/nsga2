from problem import zdt

test_problem = ["ZDT1", "ZDT2"]

problems = {
    "ZDT1": zdt.ZDT1,
}

chosen_problem = test_problem[0]

pop_size = 100
max_gen = 100
max_run = 20
mutation_pro = 1 / problems[chosen_problem]().d
crossover_pro = 0.9

run=1

