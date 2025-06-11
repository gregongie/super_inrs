settings = {
    "phantom_name": "PWC_BRAIN",
    "K": 64,
    "arch": "ffrelu_shallow",
    "arch_options": {"ff_freq":"gridded", "K0":20, "width":500, "reg_type":"mod2"},
    "arch_init_wts": "ffrelu_shallow_K0_20_width_500_init.pth", #use saved init for reproduciblity
    "seed": 1,
    "nx": 1024,
    "coords_range": (0, 1),
    "lambda": 0,
    #"lambda": 0.001, #small
    #"lambda": 0.1,   #medium
    #"lambda": 5,     #large
    "epochs": 10000,
    "step_size": 10000,
    "gamma": 0.1,
    "lr": 1e-3
}