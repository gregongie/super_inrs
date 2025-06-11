settings = {
    "phantom_name": "DOTS",
    "K": 32,
    "arch": "ffrelu_shallow",
    "arch_options": {"ff_freq":"gridded", "K0":10, "width":100, "reg_type":"mod2"},
    "arch_init_wts": "ffrelu_shallow_K0_10_width_100_init.pth", #use saved init for reproduciblity
    "seed": 1,
    "nx": 1024,
    "coords_range": (0, 1),
    "lambda": 1.0,
    "epochs": 50000,
    "step_size": 40000,
    "gamma": 0.1,
    "lr": 1e-3
}