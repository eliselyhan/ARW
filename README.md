# Model Assessment and Selection under Temporal Distribution Shift

Paper: Han, E., Huang, C. and Wang, K., 2024. Model Assessment and Selection under Temporal Distribution Shift. [arXiv preprint arXiv:2402.08672](https://arxiv.org/abs/2402.08672). To appear in ICML 2024.

## Python implementations

The file <a href="./code-synthetic-data/ARW.py">`ARW.py`</a> provides Python implementations of Adaptive Rolling Window (ARW) methods for model assessment and selection.

1. Function `ARWME`: Adaptive Rolling Window for Mean Estimation (Algorithm 1 in the paper).
  
2. Function `tournament_selection`: Single-Elimination Tournament for Model Selection (Algorithm 3 in the paper).

## Experiments in the paper

The folders `code-synthetic-data`, `code-arxiv`, and `code-housing` contain all the code for reproducing the experimental results.


## Citation
```
@article{HHW24,
  title={Model Assessment and Selection under Temporal Distribution Shift},
  author={Han, Elise and Huang, Chengpiao and Wang, Kaizheng},
  journal={arXiv preprint arXiv:2402.08672},
  year={2024}
}
```
