# 🔴 Project 1: Optimization
## Comparison of Optimization Algorithms for Hyperparameter Tuning in Steel Fault Classification

---

## 📌 Project Information

| Item | Description |
|------|-------------|
| **Course** | Optimization |
| **Topic** | Hyperparameter Optimization |
| **Dataset** | Steel Plates Fault (OpenML ID: 1504) |
| **Level** | Master's Degree |

---

## 🎯 Objectives

1. Compare different hyperparameter optimization methods
2. Analyze convergence and efficiency of algorithms
3. Find the best settings for machine learning models
4. Evaluate time complexity of each method

---

## 📊 Dataset

- **Source:** [OpenML (ID: 1504)](https://www.openml.org/d/1504)
- **Samples:** 1,941 steel plate faults
- **Features:** 27 numerical features
- **Classes:** 7 fault types

---

## 🛠️ Methods

| # | Method | Library |
|---|--------|---------|
| 1 | Grid Search | `sklearn.GridSearchCV` |
| 2 | Random Search | `sklearn.RandomizedSearchCV` |
| 3 | Bayesian Optimization | `Optuna` |

---

## 📁 Project Structure

```
Project_1_Optimization/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── optimization.py
├── figures/
├── report/
├── presentation/
├── main.py
├── requirements.txt
├── README_EN.md
├── README_FA.md
└── README_TR.md
```

---

## 🚀 Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
```

---

## 📈 Expected Outputs

- Optimization algorithms comparison table
- Convergence plots
- Runtime comparison chart
- Best hyperparameters for SVM, Random Forest, Neural Network
- Sensitivity analysis

---

## 📚 References

1. Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization.
2. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization.
3. Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework.

