# 🟢 Project 2: Machine Learning & Pattern Recognition
## Multi-class Classification of Steel Plate Faults using Machine Learning Algorithms

---

## 📌 Project Information

| Item | Description |
|------|-------------|
| **Course** | Machine Learning & Pattern Recognition |
| **Topic** | Multi-class Classification |
| **Dataset** | Steel Plates Fault (OpenML ID: 1504) |
| **Level** | Master's Degree |

---

## 🎯 Objectives

1. Compare different classification algorithms
2. Perform feature selection to identify important features
3. Evaluate model performance using various metrics
4. Analyze confusion matrix and ROC curves

---

## 📊 Dataset

- **Source:** [OpenML (ID: 1504)](https://www.openml.org/d/1504)
- **Samples:** 1,941 steel plate faults
- **Features:** 27 numerical features
- **Classes:** 7 fault types (Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps, Other)

---

## 🛠️ Models

| # | Model | Type |
|---|-------|------|
| 1 | Logistic Regression | Linear |
| 2 | K-Nearest Neighbors | Distance-based |
| 3 | Decision Tree | Tree-based |
| 4 | Random Forest | Ensemble |
| 5 | Gradient Boosting | Ensemble |
| 6 | XGBoost | Ensemble |
| 7 | SVM (RBF) | Kernel-based |
| 8 | Neural Network (MLP) | Deep Learning |

---

## 📁 Project Structure

```
Project_2_MachineLearning/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   └── visualization.py
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
pip install -r requirements.txt
python main.py
```

---

## 📈 Expected Outputs

- Model comparison table (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrix for best model
- ROC curves for all models
- Feature importance plot
- Classification report

---

## 📚 References

1. Scikit-learn Documentation
2. XGBoost Documentation
3. Bishop, C. M. (2006). Pattern Recognition and Machine Learning

