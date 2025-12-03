# 🔵 Project 3: Data Mining
## Pattern Discovery and Knowledge Extraction from Steel Plates Fault Data

---

## 📌 Project Information

| Item | Description |
|------|-------------|
| **Course** | Data Mining |
| **Topic** | Pattern Discovery & Clustering |
| **Dataset** | Steel Plates Fault (OpenML ID: 1504) |
| **Level** | Master's Degree |

---

## 🎯 Objectives

1. Perform comprehensive Exploratory Data Analysis (EDA)
2. Discover hidden patterns and relationships in data
3. Apply dimensionality reduction techniques (PCA, t-SNE)
4. Perform clustering and compare with actual labels
5. Detect anomalies in the data

---

## 📊 Dataset

- **Source:** [OpenML (ID: 1504)](https://www.openml.org/d/1504)
- **Samples:** 1,941 steel plate faults
- **Features:** 27 numerical features
- **Classes:** 7 fault types

---

## 🛠️ Techniques

| # | Technique | Purpose |
|---|-----------|---------|
| 1 | Correlation Analysis | Feature relationships |
| 2 | PCA | Dimensionality reduction |
| 3 | t-SNE | Visualization |
| 4 | K-Means | Partitioning clustering |
| 5 | Hierarchical Clustering | Dendrogram analysis |
| 6 | DBSCAN | Density-based clustering |
| 7 | Isolation Forest | Anomaly detection |

---

## 📁 Project Structure

```
Project_3_DataMining/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── mining.py
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

- Correlation heatmap
- Class distribution chart
- PCA visualization (2D, 3D)
- t-SNE visualization
- Elbow plot for optimal K
- Clustering dendrogram
- Clustering evaluation metrics (ARI, NMI)
- Anomaly detection results

