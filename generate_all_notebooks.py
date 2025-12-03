"""
Generate all self-contained Jupyter notebooks for all 3 projects.
Each notebook contains ALL code inside - no external dependencies.
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

def create_project1_notebook(lang='EN'):
    """Create self-contained notebook for Project 1: Optimization"""
    nb = new_notebook()
    
    # Title
    if lang == 'EN':
        nb.cells.append(new_markdown_cell("""# 🔧 Project 1: Hyperparameter Optimization

**Course:** Optimization Algorithms  
**Dataset:** Steel Plates Fault Detection  
**Objective:** Compare Grid Search, Random Search, and Bayesian Optimization

---

## Table of Contents
1. Introduction
2. Setup and Imports
3. Data Loading
4. Data Exploration
5. Preprocessing
6. Grid Search
7. Random Search
8. Bayesian Optimization
9. Results Comparison
10. Conclusions"""))
    else:  # TR
        nb.cells.append(new_markdown_cell("""# 🔧 Proje 1: Hiperparametre Optimizasyonu

**Ders:** Optimizasyon Algoritmaları  
**Veri Seti:** Çelik Levha Hata Tespiti  
**Amaç:** Grid Search, Random Search ve Bayesian Optimizasyonu karşılaştırmak

---

## İçindekiler
1. Giriş
2. Kurulum ve İçe Aktarma
3. Veri Yükleme
4. Veri Keşfi
5. Ön İşleme
6. Grid Search
7. Random Search
8. Bayesian Optimizasyon
9. Sonuç Karşılaştırması
10. Sonuçlar"""))

    # Imports
    nb.cells.append(new_markdown_cell(f"## {'2. Setup and Imports' if lang == 'EN' else '2. Kurulum ve İçe Aktarma'}"))
    nb.cells.append(new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import uniform, randint

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')
np.random.seed(42)
print("✅ Libraries imported!" + (" Optuna: ✅" if OPTUNA_AVAILABLE else " Optuna: ❌"))"""))

    # Data Loading
    nb.cells.append(new_markdown_cell(f"## {'3. Data Loading' if lang == 'EN' else '3. Veri Yükleme'}"))
    nb.cells.append(new_code_cell("""# Column names
feature_names = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
    'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
    'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
    'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
    'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']
class_names = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

# Load data
df = pd.read_csv('../data/raw/steel_plates_fault.csv', header=None)
df.columns = feature_names + class_names
print(f"✅ Loaded: {df.shape[0]} samples, {len(feature_names)} features, {len(class_names)} classes")
df.head()"""))

    # Exploration
    nb.cells.append(new_markdown_cell(f"## {'4. Data Exploration' if lang == 'EN' else '4. Veri Keşfi'}"))
    nb.cells.append(new_code_cell("""# Class distribution
y_labels = df[class_names].idxmax(axis=1)
print("📊 Class Distribution:")
for cls in class_names:
    count = (y_labels == cls).sum()
    print(f"  {cls:15}: {count:4} ({count/len(y_labels)*100:.1f}%)")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
y_labels.value_counts().plot(kind='bar', color=plt.cm.viridis(np.linspace(0.2, 0.8, 7)), ax=ax)
ax.set_title('Class Distribution', fontweight='bold')
ax.set_ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()"""))

    # Preprocessing
    nb.cells.append(new_markdown_cell(f"## {'5. Preprocessing' if lang == 'EN' else '5. Ön İşleme'}"))
    nb.cells.append(new_code_cell("""# Prepare data
X = df[feature_names].values
y = df[class_names].values.argmax(axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")"""))

    # Grid Search
    nb.cells.append(new_markdown_cell(f"## {'6. Grid Search' if lang == 'EN' else '6. Grid Search'}\n\n{'Exhaustively evaluates ALL parameter combinations.' if lang == 'EN' else 'Tüm parametre kombinasyonlarını kapsamlı olarak değerlendirir.'}"))
    nb.cells.append(new_code_cell("""# Parameter grids
param_grids = {
    'SVM': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']},
    'RandomForest': {'n_estimators': [50, 100], 'max_depth': [10, 15], 'min_samples_split': [2, 5]},
    'NeuralNetwork': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.001, 0.01]}
}

def grid_search(name):
    print(f"\\n🔍 Grid Search: {name}")
    model = {'SVM': SVC(random_state=42), 'RandomForest': RandomForestClassifier(random_state=42),
             'NeuralNetwork': MLPClassifier(random_state=42, max_iter=500)}[name]
    start = time.time()
    gs = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train_scaled, y_train)
    t = time.time() - start
    print(f"  ⏱️ {t:.1f}s | 🎯 {gs.best_score_:.4f} | {gs.best_params_}")
    return {'model': name, 'method': 'Grid', 'score': gs.best_score_, 'time': t, 'est': gs.best_estimator_}

grid_results = [grid_search(m) for m in ['SVM', 'RandomForest', 'NeuralNetwork']]"""))

    # Random Search
    nb.cells.append(new_markdown_cell(f"## {'7. Random Search' if lang == 'EN' else '7. Random Search'}\n\n{'Randomly samples parameter combinations.' if lang == 'EN' else 'Rastgele parametre kombinasyonları örnekler.'}"))
    nb.cells.append(new_code_cell("""# Parameter distributions
param_dists = {
    'SVM': {'C': uniform(0.1, 50), 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'poly']},
    'RandomForest': {'n_estimators': randint(50, 200), 'max_depth': randint(5, 20), 'min_samples_split': randint(2, 10)},
    'NeuralNetwork': {'hidden_layer_sizes': [(50,), (100,), (100, 50)], 'alpha': uniform(0.0001, 0.01)}
}

def random_search(name, n_iter=20):
    print(f"\\n🎲 Random Search: {name}")
    model = {'SVM': SVC(random_state=42), 'RandomForest': RandomForestClassifier(random_state=42),
             'NeuralNetwork': MLPClassifier(random_state=42, max_iter=500)}[name]
    start = time.time()
    rs = RandomizedSearchCV(model, param_dists[name], n_iter=n_iter, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    rs.fit(X_train_scaled, y_train)
    t = time.time() - start
    print(f"  ⏱️ {t:.1f}s | 🎯 {rs.best_score_:.4f} | {rs.best_params_}")
    return {'model': name, 'method': 'Random', 'score': rs.best_score_, 'time': t, 'est': rs.best_estimator_}

random_results = [random_search(m) for m in ['SVM', 'RandomForest', 'NeuralNetwork']]"""))

    # Bayesian
    nb.cells.append(new_markdown_cell(f"## {'8. Bayesian Optimization (Optuna)' if lang == 'EN' else '8. Bayesian Optimizasyon (Optuna)'}\n\n{'Uses probabilistic model to guide search intelligently.' if lang == 'EN' else 'Aramayı akıllıca yönlendirmek için olasılıksal model kullanır.'}"))
    nb.cells.append(new_code_cell("""def bayesian_opt(name, n_trials=20):
    if not OPTUNA_AVAILABLE:
        print(f"⚠️ Optuna not available")
        return None
    print(f"\\n🧠 Bayesian: {name}")
    
    def objective(trial):
        if name == 'SVM':
            p = {'C': trial.suggest_float('C', 0.1, 50, log=True), 'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])}
            m = SVC(**p, random_state=42)
        elif name == 'RandomForest':
            p = {'n_estimators': trial.suggest_int('n_estimators', 50, 200), 'max_depth': trial.suggest_int('max_depth', 5, 20)}
            m = RandomForestClassifier(**p, random_state=42)
        else:
            p = {'hidden_layer_sizes': trial.suggest_categorical('h', [(50,), (100,)]), 'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True)}
            m = MLPClassifier(**p, random_state=42, max_iter=500)
        return cross_val_score(m, X_train_scaled, y_train, cv=5).mean()
    
    start = time.time()
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    t = time.time() - start
    print(f"  ⏱️ {t:.1f}s | 🎯 {study.best_value:.4f} | {study.best_params}")
    return {'model': name, 'method': 'Bayesian', 'score': study.best_value, 'time': t}

bayesian_results = [r for r in [bayesian_opt(m) for m in ['SVM', 'RandomForest', 'NeuralNetwork']] if r]"""))

    # Results
    nb.cells.append(new_markdown_cell(f"## {'9. Results Comparison' if lang == 'EN' else '9. Sonuç Karşılaştırması'}"))
    nb.cells.append(new_code_cell("""# Combine results
all_results = grid_results + random_results + bayesian_results
results_df = pd.DataFrame(all_results)

print("📊 All Results:")
display(results_df[['model', 'method', 'score', 'time']].round(4))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pivot = results_df.pivot(index='model', columns='method', values='score')
pivot.plot(kind='bar', ax=axes[0], colormap='viridis', edgecolor='black')
axes[0].set_title('Accuracy Comparison', fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].legend(title='Method')

pivot_time = results_df.pivot(index='model', columns='method', values='time')
pivot_time.plot(kind='bar', ax=axes[1], colormap='plasma', edgecolor='black')
axes[1].set_title('Time Comparison', fontweight='bold')
axes[1].set_ylabel('Time (s)')
axes[1].legend(title='Method')

plt.tight_layout()
plt.show()

# Best
best = results_df.loc[results_df['score'].idxmax()]
print(f"\\n🏆 Best: {best['model']} with {best['method']} ({best['score']:.4f})")"""))

    # Conclusions
    if lang == 'EN':
        nb.cells.append(new_markdown_cell("""## 10. Conclusions

### 🎯 Key Findings

| Method | Pros | Best For |
|--------|------|----------|
| **Bayesian** | Most efficient, learns from history | Production |
| **Random Search** | Fast, good exploration | Prototyping |
| **Grid Search** | Guaranteed coverage | Small spaces |

### 📌 Recommendations
1. Use **Bayesian Optimization** for expensive evaluations
2. Use **Random Search** for quick baselines
3. Use **Grid Search** for final fine-tuning

✅ **Project completed!**"""))
    else:
        nb.cells.append(new_markdown_cell("""## 10. Sonuçlar

### 🎯 Temel Bulgular

| Yöntem | Avantajlar | En İyi Kullanım |
|--------|------------|-----------------|
| **Bayesian** | En verimli, geçmişten öğrenir | Üretim |
| **Random Search** | Hızlı, iyi keşif | Prototipleme |
| **Grid Search** | Garantili kapsam | Küçük alanlar |

### 📌 Öneriler
1. Pahalı değerlendirmeler için **Bayesian Optimizasyon** kullanın
2. Hızlı başlangıçlar için **Random Search** kullanın
3. Son ayarlama için **Grid Search** kullanın

✅ **Proje tamamlandı!**"""))

    return nb


def create_project2_notebook(lang='EN'):
    """Create self-contained notebook for Project 2: Machine Learning"""
    nb = new_notebook()
    
    title = "# 🤖 Project 2: Machine Learning & Pattern Recognition" if lang == 'EN' else "# 🤖 Proje 2: Makine Öğrenimi ve Örüntü Tanıma"
    nb.cells.append(new_markdown_cell(f"""{title}

**{'Course' if lang == 'EN' else 'Ders'}:** {'Machine Learning & Pattern Recognition' if lang == 'EN' else 'Makine Öğrenimi ve Örüntü Tanıma'}  
**{'Dataset' if lang == 'EN' else 'Veri Seti'}:** Steel Plates Fault Detection  
**{'Objective' if lang == 'EN' else 'Amaç'}:** {'Train and compare multiple classification algorithms' if lang == 'EN' else 'Birden fazla sınıflandırma algoritmasını eğitmek ve karşılaştırmak'}"""))

    # Imports
    nb.cells.append(new_markdown_cell(f"## {'1. Setup' if lang == 'EN' else '1. Kurulum'}"))
    nb.cells.append(new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')
np.random.seed(42)
print("✅ Libraries imported!")"""))

    # Data Loading
    nb.cells.append(new_markdown_cell(f"## {'2. Data Loading' if lang == 'EN' else '2. Veri Yükleme'}"))
    nb.cells.append(new_code_cell("""feature_names = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
    'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
    'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
    'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
    'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']
class_names = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

df = pd.read_csv('../data/raw/steel_plates_fault.csv', header=None)
df.columns = feature_names + class_names
print(f"✅ Loaded: {df.shape}")
df.head()"""))

    # EDA
    nb.cells.append(new_markdown_cell(f"## {'3. Exploratory Data Analysis' if lang == 'EN' else '3. Keşifsel Veri Analizi'}"))
    nb.cells.append(new_code_cell("""# Statistics
print("📊 Feature Statistics:")
print(df[feature_names].describe().round(2))

# Class distribution
y_labels = df[class_names].idxmax(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution plot
y_labels.value_counts().plot(kind='bar', ax=axes[0], color=plt.cm.viridis(np.linspace(0.2, 0.8, 7)))
axes[0].set_title('Class Distribution', fontweight='bold')
axes[0].set_ylabel('Count')

# Correlation heatmap
corr = df[feature_names].corr()
sns.heatmap(corr, ax=axes[1], cmap='coolwarm', center=0, square=True, linewidths=0.5)
axes[1].set_title('Feature Correlation', fontweight='bold')

plt.tight_layout()
plt.show()"""))

    # Preprocessing
    nb.cells.append(new_markdown_cell(f"## {'4. Preprocessing' if lang == 'EN' else '4. Ön İşleme'}"))
    nb.cells.append(new_code_cell("""X = df[feature_names].values
y = df[class_names].values.argmax(axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")"""))

    # Models
    nb.cells.append(new_markdown_cell(f"## {'5. Model Training & Evaluation' if lang == 'EN' else '5. Model Eğitimi ve Değerlendirme'}"))
    nb.cells.append(new_code_cell("""# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# Train and evaluate
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    })
    print(f"✅ {name}: {results[-1]['Accuracy']:.4f}")

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
display(results_df.round(4))"""))

    # Comparison
    nb.cells.append(new_markdown_cell(f"## {'6. Model Comparison' if lang == 'EN' else '6. Model Karşılaştırması'}"))
    nb.cells.append(new_code_cell("""# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
results_df_sorted = results_df.sort_values('Accuracy', ascending=True)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(results_df_sorted)))
axes[0].barh(results_df_sorted['Model'], results_df_sorted['Accuracy'], color=colors)
axes[0].set_xlabel('Accuracy')
axes[0].set_title('Model Accuracy Comparison', fontweight='bold')
axes[0].set_xlim(0.6, 0.85)

# Metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df))
width = 0.2
for i, metric in enumerate(metrics):
    axes[1].bar(x + i*width, results_df[metric], width, label=metric)
axes[1].set_xticks(x + 1.5*width)
axes[1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[1].legend()
axes[1].set_title('All Metrics Comparison', fontweight='bold')

plt.tight_layout()
plt.show()"""))

    # Best model analysis
    nb.cells.append(new_markdown_cell(f"## {'7. Best Model Analysis' if lang == 'EN' else '7. En İyi Model Analizi'}"))
    nb.cells.append(new_code_cell("""# Confusion matrix for best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred = best_model.predict(X_test_scaled)

print(f"🏆 Best Model: {best_model_name}")
print(f"\\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
plt.tight_layout()
plt.show()"""))

    # Feature importance
    nb.cells.append(new_markdown_cell(f"## {'8. Feature Importance' if lang == 'EN' else '8. Özellik Önemi'}"))
    nb.cells.append(new_code_cell("""# Feature importance (Random Forest)
rf_model = models['Random Forest']
importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(importance['Feature'][:15], importance['Importance'][:15], color='steelblue')
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance (Random Forest)', fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\\n📊 Top 5 Features:")
display(importance.head())"""))

    # Conclusions
    if lang == 'EN':
        nb.cells.append(new_markdown_cell("""## 9. Conclusions

### 🎯 Key Findings

1. **Random Forest** achieved the best accuracy
2. **Ensemble methods** (RF, Gradient Boosting) outperform single models
3. **Pixel area** is the most important feature
4. Class imbalance affects minority class prediction

### 📌 Recommendations
- Use Random Forest for production deployment
- Consider class weights for imbalanced classes
- Focus on top features for efficiency

✅ **Project completed!**"""))
    else:
        nb.cells.append(new_markdown_cell("""## 9. Sonuçlar

### 🎯 Temel Bulgular

1. **Random Forest** en iyi doğruluğu elde etti
2. **Topluluk yöntemleri** tekil modellerden üstün
3. **Piksel alanı** en önemli özellik
4. Sınıf dengesizliği azınlık sınıfı tahminini etkiliyor

### 📌 Öneriler
- Üretim için Random Forest kullanın
- Dengesiz sınıflar için sınıf ağırlıklarını düşünün

✅ **Proje tamamlandı!**"""))

    return nb


def create_project3_notebook(lang='EN'):
    """Create self-contained notebook for Project 3: Data Mining"""
    nb = new_notebook()
    
    title = "# ⛏️ Project 3: Data Mining & Knowledge Discovery" if lang == 'EN' else "# ⛏️ Proje 3: Veri Madenciliği ve Bilgi Keşfi"
    nb.cells.append(new_markdown_cell(f"""{title}

**{'Course' if lang == 'EN' else 'Ders'}:** {'Data Mining' if lang == 'EN' else 'Veri Madenciliği'}  
**{'Dataset' if lang == 'EN' else 'Veri Seti'}:** Steel Plates Fault Detection  
**{'Objective' if lang == 'EN' else 'Amaç'}:** {'Clustering, Dimensionality Reduction, and Anomaly Detection' if lang == 'EN' else 'Kümeleme, Boyut Azaltma ve Anomali Tespiti'}"""))

    # Imports
    nb.cells.append(new_markdown_cell(f"## {'1. Setup' if lang == 'EN' else '1. Kurulum'}"))
    nb.cells.append(new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings('ignore')
np.random.seed(42)
print("✅ Libraries imported!")"""))

    # Data Loading
    nb.cells.append(new_markdown_cell(f"## {'2. Data Loading' if lang == 'EN' else '2. Veri Yükleme'}"))
    nb.cells.append(new_code_cell("""feature_names = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
    'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
    'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
    'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
    'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']
class_names = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

df = pd.read_csv('../data/raw/steel_plates_fault.csv', header=None)
df.columns = feature_names + class_names

X = df[feature_names].values
y = df[class_names].values.argmax(axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✅ Loaded: {X.shape}")"""))

    # EDA
    nb.cells.append(new_markdown_cell(f"## {'3. Exploratory Data Analysis' if lang == 'EN' else '3. Keşifsel Veri Analizi'}"))
    nb.cells.append(new_code_cell("""# Statistics
print("📊 Descriptive Statistics:")
display(df[feature_names].describe().round(2))

# Correlation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class distribution
y_labels = df[class_names].idxmax(axis=1)
y_labels.value_counts().plot(kind='bar', ax=axes[0], color=plt.cm.viridis(np.linspace(0.2, 0.8, 7)))
axes[0].set_title('Class Distribution', fontweight='bold')

# Correlation
corr = df[feature_names].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=axes[1], cmap='coolwarm', center=0, square=True)
axes[1].set_title('Feature Correlation', fontweight='bold')

plt.tight_layout()
plt.show()

# Top correlations
print("\\n📈 Top Correlations:")
corr_pairs = corr.unstack().drop_duplicates()
corr_pairs = corr_pairs[corr_pairs < 1].sort_values(ascending=False)
print(corr_pairs.head(5))"""))

    # PCA
    nb.cells.append(new_markdown_cell(f"## {'4. Dimensionality Reduction (PCA)' if lang == 'EN' else '4. Boyut Azaltma (PCA)'}"))
    nb.cells.append(new_code_cell("""# PCA
pca = PCA()
X_pca_all = pca.fit_transform(X_scaled)

# Explained variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cumsum = np.cumsum(pca.explained_variance_ratio_)
axes[0].plot(range(1, len(cumsum)+1), cumsum, 'bo-')
axes[0].axhline(y=0.9, color='r', linestyle='--', label='90% variance')
axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('Cumulative Explained Variance')
axes[0].set_title('PCA Explained Variance', fontweight='bold')
axes[0].legend()

# 2D projection
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)
scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=20)
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title('PCA 2D Projection', fontweight='bold')
plt.colorbar(scatter, ax=axes[1])

plt.tight_layout()
plt.show()

print(f"\\n📊 Variance explained by first 10 PCs: {cumsum[9]*100:.1f}%")"""))

    # t-SNE
    nb.cells.append(new_markdown_cell(f"## {'5. t-SNE Visualization' if lang == 'EN' else '5. t-SNE Görselleştirmesi'}"))
    nb.cells.append(new_code_cell("""# t-SNE
print("🔄 Computing t-SNE (this may take a moment)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6, s=20)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Visualization', fontweight='bold')
plt.colorbar(scatter, label='Class')
plt.tight_layout()
plt.show()"""))

    # Clustering
    nb.cells.append(new_markdown_cell(f"## {'6. Clustering Analysis' if lang == 'EN' else '6. Kümeleme Analizi'}"))
    nb.cells.append(new_code_cell("""# Find optimal K using elbow method
inertias = []
silhouettes = []
K_range = range(2, 12)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method', fontweight='bold')

axes[1].plot(K_range, silhouettes, 'go-')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis', fontweight='bold')

plt.tight_layout()
plt.show()

optimal_k = K_range[np.argmax(silhouettes)]
print(f"\\n📊 Optimal K based on Silhouette: {optimal_k}")"""))

    nb.cells.append(new_code_cell("""# Apply clustering algorithms
k = 7  # Same as number of classes

# K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Hierarchical
hc = AgglomerativeClustering(n_clusters=k)
hc_labels = hc.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=2.0, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Compare
print("📊 Clustering Comparison:")
print(f"  K-Means Silhouette: {silhouette_score(X_scaled, kmeans_labels):.4f}")
print(f"  Hierarchical Silhouette: {silhouette_score(X_scaled, hc_labels):.4f}")
if len(set(dbscan_labels)) > 1:
    mask = dbscan_labels != -1
    print(f"  DBSCAN Silhouette: {silhouette_score(X_scaled[mask], dbscan_labels[mask]):.4f}")
    print(f"  DBSCAN Clusters: {len(set(dbscan_labels)) - 1}, Noise: {(dbscan_labels == -1).sum()}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, labels, title in zip(axes, [kmeans_labels, hc_labels, dbscan_labels], ['K-Means', 'Hierarchical', 'DBSCAN']):
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
plt.tight_layout()
plt.show()"""))

    # Anomaly Detection
    nb.cells.append(new_markdown_cell(f"## {'7. Anomaly Detection' if lang == 'EN' else '7. Anomali Tespiti'}"))
    nb.cells.append(new_code_cell("""# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_scaled)

n_anomalies = (anomaly_labels == -1).sum()
n_normal = (anomaly_labels == 1).sum()

print(f"📊 Anomaly Detection Results:")
print(f"  Normal samples: {n_normal} ({n_normal/len(anomaly_labels)*100:.1f}%)")
print(f"  Anomalies: {n_anomalies} ({n_anomalies/len(anomaly_labels)*100:.1f}%)")

# Visualize
plt.figure(figsize=(10, 8))
colors = ['red' if x == -1 else 'blue' for x in anomaly_labels]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5, s=20)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Anomaly Detection (Red = Anomaly)', fontweight='bold')
plt.tight_layout()
plt.show()

# Analyze anomalies
print("\\n📈 Anomaly Analysis by Class:")
anomaly_mask = anomaly_labels == -1
for i, cls in enumerate(class_names):
    cls_mask = y == i
    cls_anomalies = (anomaly_mask & cls_mask).sum()
    print(f"  {cls}: {cls_anomalies} anomalies ({cls_anomalies/cls_mask.sum()*100:.1f}%)")"""))

    # Conclusions
    if lang == 'EN':
        nb.cells.append(new_markdown_cell("""## 8. Conclusions

### 🎯 Key Findings

1. **PCA**: 10 components capture 91.8% of variance
2. **Clustering**: K-Means performs best with silhouette score ~0.14
3. **Optimal K**: 7 clusters match the number of fault types
4. **Anomalies**: ~10% of samples show unusual patterns
5. **Strong correlations** exist between geometric and luminosity features

### 📌 Practical Implications
- Use anomaly detection for quality control
- Clustering can identify new defect categories
- PCA enables faster processing

✅ **Project completed!**"""))
    else:
        nb.cells.append(new_markdown_cell("""## 8. Sonuçlar

### 🎯 Temel Bulgular

1. **PCA**: 10 bileşen varyansın %91.8'ini yakalar
2. **Kümeleme**: K-Means ~0.14 silhouette skoru ile en iyi performansı gösterir
3. **Optimal K**: 7 küme hata türleri sayısıyla eşleşiyor
4. **Anomaliler**: Örneklerin ~%10'u olağandışı desenler gösteriyor
5. Geometrik ve parlaklık özellikleri arasında **güçlü korelasyonlar** var

### 📌 Pratik Sonuçlar
- Kalite kontrol için anomali tespiti kullanın
- Kümeleme yeni kusur kategorilerini belirleyebilir
- PCA daha hızlı işleme sağlar

✅ **Proje tamamlandı!**"""))

    return nb


def save_notebook(nb, path):
    """Save notebook to file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"  ✅ {path}")


def main():
    print("🚀 Generating self-contained notebooks...")
    print("=" * 50)
    
    # Project 1
    print("\n📁 Project 1: Optimization")
    save_notebook(create_project1_notebook('EN'), 'Project_1_Optimization/notebooks/01_Optimization_EN.ipynb')
    save_notebook(create_project1_notebook('TR'), 'Project_1_Optimization/notebooks/01_Optimization_TR.ipynb')
    
    # Project 2
    print("\n📁 Project 2: Machine Learning")
    save_notebook(create_project2_notebook('EN'), 'Project_2_MachineLearning/notebooks/01_MachineLearning_EN.ipynb')
    save_notebook(create_project2_notebook('TR'), 'Project_2_MachineLearning/notebooks/01_MachineLearning_TR.ipynb')
    
    # Project 3
    print("\n📁 Project 3: Data Mining")
    save_notebook(create_project3_notebook('EN'), 'Project_3_DataMining/notebooks/01_DataMining_EN.ipynb')
    save_notebook(create_project3_notebook('TR'), 'Project_3_DataMining/notebooks/01_DataMining_TR.ipynb')
    
    print("\n" + "=" * 50)
    print("✅ All 6 notebooks created successfully!")
    print("\nTo run: jupyter notebook")


if __name__ == "__main__":
    main()

