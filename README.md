# adaptive-KTE


# For reproducibility

1) Create conda environment with the following command

```conda create -n adaptive_exp python=3.8```

2) Activate the conda environment

```conda activate adaptive_exp```

3) Install the required libraries with the command

```pip install -r requirements.txt```

4) To reproduce the results given in the paper, the corresponding python scripts needs to be run:
    * To reproduce Figure 1:
        ```python experiment_failure_KTE_CLT.py```
    * To reproduce Figure 2 and 3:
        ```python experiment_adaptive_with-cosine-structural-func.py```
    * To reproduce Table 1, Figure 10 and Figure 11:
        ```python experiment_adaptive-IHDP-sinusoidal_structural_func.py```
    * To reproduce Table 2: (dsprite experiments may take a long time. One can utilize the bash file to paralelize the experiments in a SLURM. See dsprite.sh)
        ```python experiments_dsprite.py```
    * To reproduce Figure 6 and 7:
        ```python experiment_adaptive-linear-structural-func.py```
    * To reproduce Figure 8 and 9:
        ```python experiment_adaptive-sigmoidal-structural-func.py```
