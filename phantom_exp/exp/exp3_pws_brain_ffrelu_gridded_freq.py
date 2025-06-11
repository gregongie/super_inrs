settings = {
    "phantom_name": "PWS_BRAIN",
    "K": 64,
    "arch": "ffrelu_deep",
    "arch_options": {"ff_freq":"gridded", "K0":10, "width":100, "depth":20},
    "seed": 1,
    "nx": 1024,
    "coords_range": (0, 1),
    "lambda": 1e-11,
    "epochs": 50000,
    "step_size": 40000,
    "gamma": 0.1,
    "lr": 1e-3
}