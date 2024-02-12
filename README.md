# Model Assessment and Selection under Temporal Distribution Shift
This is the code associated with the paper [Model Assessment and Selection under Temporal Distribution Shift].

## File Overview `./code_syn`
### 1. <a href="./code_syn/ARW.py">`ARW.py`</a>
- Provides functions that implement the Single-Elimination Tournament for Model Selection (Algorithm 3 in the paper) via Adaptive Rolling Window(ARW)

### 2. <a href="./code_syn/algo_syn">`algo_syn.py`</a>
- Provides algorithms for training with fixed window sizes, model selection with fixed window sizes, and model selection via Adaptive Rolling Window(ARW)
  
### 3. <a href="./code_syn/generate_syn_data.py">`generate_syn_data.py`</a>
- Provides functions for generating synthetic data

### 4. <a href="./code_syn/figures.ipynb">`figures.ipynb`</a>
- Provides an example of code usage using synthetic data and produces figures presented in the paper

## Results
Figures using synthetic data presented in the paper can be reproduced using <a href="./code_syn/figures.ipynb">`figures.ipynb`</a>.
