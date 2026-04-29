# On the Alignment between Fairness and Accuracy: from the Perspective of Adversarial Robustness

While numerous work has been proposed to address fairness in machine learning, existing methods do not guarantee fair predictions under imperceptible feature perturbation, and a seemingly fair model can suffer from large group-wise disparities under such perturbation. Moreover, while adversarial training has been shown to be reliable in improving a model’s robustness to defend against adversarial feature perturbation that deteriorates accuracy, it has not been properly studied in the context of adversarial perturbation against fairness. To tackle these challenges, in this paper, we study the problem of adversarial attack and adversarial robustness w.r.t. two terms: fairness and accuracy. From the adversarial attack perspective, we propose a unified structure for adversarial attacks against fairness which brings together common notions in group fairness, and we theoretically prove the equivalence of adversarial attacks against different fairness notions. Further, we derive the connections between adversarial attacks against fairness and those against accuracy. From the adversarial robustness perspective, we theoretically align robustness to adversarial attacks against fairness and accuracy, where robustness w.r.t. one term enhances robustness w.r.t. the other term. Our study suggests a novel way to unify adversarial training w.r.t. fairness and accuracy, and experiments show our proposed method achieves better robustness w.r.t. both terms.

# Repository Contents

This repository contains implementations of fair adversarial training methods on two datasets:
- **adult.ipynb** - Implementation on Adult Income dataset
- **compas.ipynb** - Implementation on COMPAS Recidivism dataset
- **adult_reconstruction.csv** - Adult dataset 
- **compas-scores-two-years.csv** - COMPAS dataset
- **requirements.txt** - Python package dependencies 

# Setup Instructions

This code can be run in a Google Colab environment or locally on your machine. Below are the instructions for both setups.

## Running in Google Colab
1. Open the desired notebook (e.g., `adult.ipynb`) in Google Colab.
2. Add the dataset files (`adult_reconstruction.csv` and `compas-scores-two-years.csv`) to your Google Drive and update the file paths in the notebook accordingly. For example:
   ```python
   data = pd.read_csv('./adult_reconstruction.csv')
   ```
3. Follow the instructions in the notebook to install required packages and run the code cells sequentially.

## Running Locally

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step-by-Step Installation Guide

**Step 1: Clone the repository**
```bash
git clone https://github.com/cjy24/fair-adversarial-training.git
cd fair-adversarial-training
```

**Step 2: Create a virtual environment (recommended)**
```bash
python -m venv venv
```

**Step 3: Activate the virtual environment**
- **On Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **On macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

**Step 4: Install required packages**
```bash
pip install -r requirements.txt
```

This will install all dependencies with the following specific versions:
- Pandas == 1.5.3
- Numpy == 1.23.5
- sklearn == 1.2.2
- torch == 2.0.1+cu118
- scipy == 1.10.1

**Step 5: Verify installation**
```bash
python -c "import pandas; import torch; import sklearn; print('Installation successful!')"
```

## Usage Instructions

### Running the Adult Dataset Experiment

1. Open `adult.ipynb` in Jupyter Notebook:
   ```bash
   jupyter notebook adult.ipynb
   ```

2. The notebook performs the following operations:
   - Loads and preprocesses the Adult Income dataset
   - Applies one-hot encoding to categorical variables
   - Normalizes numerical features
   - Filters data for specific demographic groups
   - Implements adversarial training on fairness metrics
   - Evaluates model robustness and fairness

3. **Note:** Ensure the file path in the notebook points to your `adult_reconstruction.csv` location. Update if needed:
   ```python
   data = pd.read_csv('./adult_reconstruction.csv')
   ```

### Running the COMPAS Dataset Experiment

1. Open `compas.ipynb` in Jupyter Notebook:
   ```bash
   jupyter notebook compas.ipynb
   ```

2. The notebook performs similar operations on the COMPAS Recidivism dataset
3. **Note:** Similarly verify the file path for `compas-scores-two-years.csv`

### Deactivating Virtual Environment

When finished, deactivate the virtual environment:
```bash
deactivate
```
## Code Structure Overview

### adult.ipynb
- **Cell 1:** Data loading and preprocessing (imports, data cleaning, normalization)
- **Cell 2:** Attack functions (PGD and other adversarial attacks)
- **Cell 3:** Model training and robustness evaluation
- **Cell 4-8:** Experiments and visualization of results
- **Cell 9-10:** Novel Implementation of Multi-attribute attack and analysis

### compas.ipynb
- Similar structure with COMPAS-specific data preprocessing and analysis

## Troubleshooting

**Issue:** "ModuleNotFoundError" when running notebooks
- **Solution:** Ensure virtual environment is activated and all packages are installed via `pip install -r requirements.txt`

**Issue:** CUDA-related errors
- **Solution:** Install CPU-only PyTorch version (see Dependencies section above)

**Issue:** CSV files not found
- **Solution:** Verify that `adult_reconstruction.csv` and `compas-scores-two-years.csv` are in the repository root directory. Update file paths in notebooks if needed.

## Citation

Code repository is forked from the original implementation by the authors of the paper "On the Alignment between Fairness and Accuracy: from the Perspective of Adversarial Robustness", published by Chengjie Yu, Yiming Ma, and Yizhou Sun in KDD 2024. 

Original code repository: https://github.com/cjy24/fair-adversarial-training
