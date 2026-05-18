# adaptive-KTE

This is the code for our [paper on Kernel Treatment Effects with Adaptively Collected Data.](https://arxiv.org/pdf/2510.10245)

Please cite our work if you find it useful for your research and work:
```
@article{zenati2025kernel,
  title={Kernel Treatment Effects with Adaptively Collected Data},
  author={Zenati, Houssam and Bozkurt, Bariscan and Gretton, Arthur},
  journal={arXiv preprint arXiv:2510.10245},
  year={2025}
}
```

## Canonical Files

- `projected_adaptive_kte.py`: implementation of the projected adaptive kernel treatment effect test.
- `baselines.py`: scalar adaptive baselines used in the paper (`CADR`, `AW-AIPW`).
- `xkte_nonadaptive.py` and `kte.py`: non-adaptive kernel baselines used in the appendix comparison.
- `dsprite_adaptive.py`: dSprite adaptive data generator.

## Paper Experiment Drivers

- `experiment_adaptive_cosine.ipynb`: main synthetic cosine calibration and power experiments.
- `experiment_adaptive_linear.ipynb`: appendix synthetic linear experiments.
- `experiment_adaptive_sigmoidal.ipynb`: appendix synthetic sigmoidal experiments.
- `experiment_adaptive_IHDP_sinusoidal.ipynb`: IHDP semi-synthetic experiments.
- `experiment_adaptive_dsprite.py`: dSprite structured-outcome experiments.
- `experiment_adaptive_rebuttal.ipynb`: adaptive comparison with non-adaptive kernel baselines.
- `experiment_almost_iid_rebuttal.ipynb`: i.i.d. comparison with non-adaptive kernel baselines.
- `scripts/run_failure_example_dr_xkte.py`: failure figure for the naive adaptive DR-xKTE statistic.
- `scripts/plot_dsprite_examples.py`: dSprite observational and counterfactual example images.

The notebooks above use the adaptive estimator and the alternating split described in the experimental appendix.

## Environment

```bash
conda create -n adaptive-kte python=3.9
conda activate adaptive-kte
pip install -r requirements.txt
```

If Matplotlib cannot write its cache, run:

```bash
export MPLCONFIGDIR="$PWD/.matplotlib"
```

## Running Experiments

Run the notebooks directly for the synthetic, IHDP, and i.i.d./adaptive comparison results.

For dSprite:

```bash
python experiment_adaptive_dsprite.py --sequential_run
python experiment_adaptive_dsprite.py --results
```

For the standalone figures:

```bash
python scripts/run_failure_example_dr_xkte.py
python scripts/plot_dsprite_examples.py
```

## Data And Outputs

- `data/ihdp.csv` is the IHDP table used by the IHDP notebook.
- `data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz` is the dSprite archive used by the image experiment.
- `figures/` contains manuscript-ready image assets.
- `results/` is generated output and is intentionally not part of the cleaned source layout.
