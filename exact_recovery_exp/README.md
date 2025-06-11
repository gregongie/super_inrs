# Exact Recovery Experiments

Scripts `thm1_table.py` and `thm2_table.py` can be used to reproduce the exact recovery proabability tables shown in **Figure 3**.

In the call to `train_student` in the main for-loop, set `std_wd = True` for standard weight decay and `std_wd = False` for modified weight decay.

In these script `job_id` is the random seed. Rerun these scripts with `job_id` set to 1,...,10 to obtain probability tables shown in Figure 3.

**CAUTION**: running one of these scripts for a single `job_id' may take several hours. We recommend running these scripts on a HPC cluster.

