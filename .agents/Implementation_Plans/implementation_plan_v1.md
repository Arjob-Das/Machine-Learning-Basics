# Implementation Plan - Modernizing Dependencies and Code Cleanup

This implementation plan details the updates to packages, module import paths, deprecated function syntax, `.gitignore`, and `README.md` across the repository.

## Proposed Changes

### Dependencies & Environment

#### [MODIFY] [requirements.txt](file:///f:/Self_Study/Machine-Learning-Basics/requirements.txt)
- Pin package versions to align with the current Python 3.12 environment.
- Add missing dependencies that are imported in various python files in the repository.
- Proposed pinned versions:
  - `numpy==1.26.4`
  - `pandas==2.2.2`
  - `scikit-learn==1.5.0`
  - `scipy==1.15.3`
  - `seaborn==0.13.2`
  - `plotly==6.0.1`
  - `folium==0.19.5`
  - `torch==2.12.0+cu130`
  - `cufflinks==0.17.3`
  - `chart-studio==1.1.0`
  - `pandas-datareader==0.10.0`
  - `nltk==3.9.1`
  - `matplotlib==3.9.0`
  - `psutil==7.0.0`
  - `tensorflow==2.16.1`

### Python Source File Updates

#### [MODIFY] [tf02.py](file:///f:/Self_Study/Machine-Learning-Basics/17-Tensor-Flow/tf02.py)
#### [MODIFY] [keras_test.py](file:///f:/Self_Study/Machine-Learning-Basics/17-Tensor-Flow/ANNs/keras_test.py)
#### [MODIFY] [keras03.py](file:///f:/Self_Study/Machine-Learning-Basics/17-Tensor-Flow/ANNs/keras03.py)
#### [MODIFY] [keras02.py](file:///f:/Self_Study/Machine-Learning-Basics/17-Tensor-Flow/ANNs/keras02.py)
#### [MODIFY] [keras01.py](file:///f:/Self_Study/Machine-Learning-Basics/17-Tensor-Flow/ANNs/keras01.py)
- Change standalone Keras imports to TensorFlow-integrated Keras imports:
  - e.g., `from keras.models import load_model` -> `from tensorflow.keras.models import load_model`
  - e.g., `from keras.models import Sequential` -> `from tensorflow.keras.models import Sequential`
  - e.g., `from keras.layers import Dense, Dropout, BatchNormalization` -> `from tensorflow.keras.layers import Dense, Dropout, BatchNormalization`
  - e.g., `from keras.optimizers import Adam` -> `from tensorflow.keras.optimizers import Adam`
  - e.g., `from keras.callbacks import EarlyStopping` -> `from tensorflow.keras.callbacks import EarlyStopping`

#### [MODIFY] [seaborn04.py](file:///f:/Self_Study/Machine-Learning-Basics/05-Seaborn/seaborn04.py)
- Replace deprecated `sns.distplot` calls with `sns.histplot` to prevent future runtime issues:
  - `g.map_diag(sns.distplot)` -> `g.map_diag(sns.histplot, kde=True)`
  - `g.map(sns.distplot, 'total_bill')` -> `g.map(sns.histplot, 'total_bill')`

### Configuration and Documentation

#### [MODIFY] [.gitignore](file:///f:/Self_Study/Machine-Learning-Basics/.gitignore)
- Add entries to ignore common Python temporary artifacts, virtual environments, local IDE metadata, and checkpoints:
  - `.venv/` and `venv/`
  - `__pycache__/` and `*.pyc`
  - `.ipynb_checkpoints/`
  - `.vscode/`, `.idea/`, `.vs/`

#### [MODIFY] [README.md](file:///f:/Self_Study/Machine-Learning-Basics/README.md)
- Update setup commands and dependency listings to match pinned requirements.
- Document compatibility notes (Python 3.12, Tensorflow 2.16+, etc.).

## Verification Plan

### Automated Tests
We will verify python import syntax compatibility by running:
- `python -c "import tensorflow; import keras; import seaborn; import plotly"`
- Dry running modified `.py` files to verify there are no compilation or runtime syntax errors.
