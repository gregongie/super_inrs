settings = {
    "phantom_name": "SL",
    "K": 48,
    "arch": "ffrelu_shallow",
    "arch_options": {"ff_freq":"gridded", "K0":10, "width":100, "reg_type":"std"},
    "arch_init_wts": "ffrelu_shallow_K0_10_width_100_init.pth", #use saved init for reproduciblity
    "seed": 1,
    "nx": 1024,
    "coords_range": (0, 1),
    "lambda": 5e-7,
    "epochs": 50000,
    "step_size": 40000,
    "gamma": 0.1,
    "lr": 1e-3
}