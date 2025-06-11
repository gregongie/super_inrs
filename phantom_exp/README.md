# Phantom Recovery Experiments

For an illustration of a basic phantom super-resolution recovery experiment see the notebook `demo.ipynb`

To reproduce the results in the paper, use the `run_exp.py` script, along with the experiment settings files in the `exp` subfolder. Each ``.py`` file in the ``exp`` subfolder corresponds to a super-resolution recovery problem for a given phantom image and INR architecture. To run one of these experiments, open a terminal in this subfolder, and use the command:

``python run_exp.py --exp [exp_name] --device [device_name]``

where ``[exp_name]`` is the name of one of the ``.py`` files in the ``exp`` subfolder (without the .py extension), and ``[device_name]`` is the device to run the experiment on, e.g., ``cuda:0``, ``cuda:1``, etc., or ``cpu``. Defaults to ``cuda:0`` (i.e., GPU 0) when ``--device`` option is not used.

For example, to run the first experiment on GPU 1 use:

`python run_exp.py --exp exp1_dot_mod_wd_1 --device cuda:1`

Results of the experiments are saved in the `results` subfolder. Each experiment generates four files:
* `[exp_name].json` is a dict containing the final metrics after training (including MSE)
* `[exp_name].npy` is the output INR image saved as a numpy array
* `[exp_name].png` is the output INR image saved in .png format for easy visualization
* `[exp_name].pth` contains the final trained INR network weights

Experiments are organized as follows:

### Filenames starting with `exp1`
Recovery of phantoms using *shallow* ReLU INRs with a Fouier features layer and various types of regularization. Results of these experiments are shown in **Figure 4** and **Figure 5** in the paper.

### Filenames starting with `exp2`
Recovery of phantoms using *deep* ReLU INRs with Fourier features layer for various depths. Results of these experiments are shown in **Figure 8** in the paper.

### Filenames starting with `exp3`
Recovery of phantoms using various popular INR architectures. Results for SL phantom are shown in **Figure 9** in the paper. Results of these experiments for the PWC BRAIN phantom are shown in **Figure 10** in the paper. Results of these experiments for the PWS BRAIN phantom are shown in the Supplementary Materials **Figure SM1**.

### Filenames starting with `unit_sizes`
Recovery of DOT and PWC BRAIN phantoms using shallow ReLU INRs with Fourier features layer for different regularization strengths. Results for the PWC BRAIN phantom are shown in **Figure 6** and **Figure 7**, and results for the DOT phantom are shown in the Supplementary Materials, **Figure SM2** and **Figure SM3**. Note: rather than duplicating these files for each regularization parameter ``lambda`` the base files provided have ``lambda`` set to 0, with the values of ``lambda`` used in the low/medium/high settings commented out.