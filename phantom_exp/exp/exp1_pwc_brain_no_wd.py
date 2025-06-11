settings = {
    "phantom_name": "PWC_BRAIN",
    "K": 64,
    "arch": "ffrelu_shallow",
    "arch_options": {"ff_freq":"gridded", "K0":20, "width":500, "reg_type":"std"},
    "arch_init_wts": "ffrelu_shallow_K0_20_width_500_init.pth", #use saved init for reproduciblity
    "seed": 1,
    "nx": 1024,
    "coords_range": (0, 1),
    "lambda": 0,
    "epochs": 50000,
    "step_size": 40000,
    "gamma": 0.1,
    "lr": 1e-3
}