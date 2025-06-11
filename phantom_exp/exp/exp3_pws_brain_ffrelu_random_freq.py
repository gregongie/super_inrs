settings = {
    "phantom_name": "PWS_BRAIN",
    "K": 64,
    "arch": "ffrelu_deep",
    "arch_options": {"ff_freq":"random", "num_freq":256, "sigma":10, "ff_seed":2, "width":100, "depth":20},
    "seed": 4,
    "nx": 1024,
    "coords_range": (0, 1),
    "lambda": 4e-8,
    "epochs": 50000,
    "step_size": 40000,
    "gamma": 0.1,
    "lr": 1e-3
}