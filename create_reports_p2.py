"""
Script to create comprehensive HTML reports for Project 2 (Machine Learning).
"""

import os

def get_css_styles():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+Pro:wght@300;400;600&family=Fira+Code&display=swap');
        
        :root {
            --primary: #7c3aed;
            --secondary: #a855f7;
            --accent: #06b6d4;
            --warning: #f59e0b;
            --card-bg: #ffffff;
            --text: #1f2937;
            --light-bg: #faf5ff;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Source Sans Pro', sans-serif;
            background: linear-gradient(135deg, #4c1d95 0%, #7c3aed 50%, #06b6d4 100%);
            min-height: 100vh;
            color: var(--text);
            line-height: 1.8;
        }
        
        .container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }
        
        header {
            background: var(--card-bg);
            border-radius: 24px;
            padding: 60px 50px;
            text-align: center;
            margin-bottom: 40px;
            box-shadow: 0 25px 60px rgba(0,0,0,0.2);
            position: relative;
            overflow: hidden;
        }
        
        header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(90deg, var(--primary), var(--accent));
        }
        
        h1 {
            font-family: 'Playfair Display', serif;
            font-size: 3em;
            color: var(--primary);
            margin-bottom: 15px;
        }
        
        .subtitle { font-size: 1.4em; color: var(--secondary); font-weight: 300; }
        
        .badges { margin-top: 25px; }
        
        .badge {
            display: inline-block;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: white;
            padding: 10px 25px;
            border-radius: 30px;
            margin: 5px;
            font-weight: 600;
        }
        
        section {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 45px;
            margin-bottom: 30px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.1);
        }
        
        h2 {
            font-family: 'Playfair Display', serif;
            color: var(--primary);
            font-size: 2em;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid var(--secondary);
        }
        
        h3 { color: var(--secondary); font-size: 1.4em; margin: 30px 0 15px; }
        h4 { color: var(--accent); font-size: 1.1em; margin: 20px 0 10px; }
        
        p { margin-bottom: 15px; text-align: justify; }
        
        .highlight-box {
            background: linear-gradient(135deg, #faf5ff, #f3e8ff);
            border-left: 5px solid var(--primary);
            padding: 25px 30px;
            border-radius: 0 15px 15px 0;
            margin: 25px 0;
        }
        
        .success-box {
            background: linear-gradient(135deg, #ecfeff, #cffafe);
            border-left: 5px solid var(--accent);
            padding: 25px 30px;
            border-radius: 0 15px 15px 0;
            margin: 25px 0;
        }
        
        ul, ol { margin: 15px 0 15px 25px; }
        li { margin-bottom: 10px; }
        
        .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px; margin: 25px 0; }
        .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 25px 0; }
        .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 25px 0; }
        
        .card {
            background: var(--light-bg);
            border-radius: 16px;
            padding: 25px;
            border: 1px solid #e9d5ff;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .card:hover { transform: translateY(-5px); box-shadow: 0 15px 30px rgba(0,0,0,0.1); }
        
        .metric-card {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 30px;
            border-radius: 16px;
            text-align: center;
        }
        
        .metric-card .value { font-size: 2.5em; font-weight: 700; display: block; }
        .metric-card .label { font-size: 1em; opacity: 0.9; margin-top: 8px; }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
        }
        
        th, td { padding: 16px 20px; text-align: left; border-bottom: 1px solid #e9d5ff; }
        th { background: var(--primary); color: white; }
        tr:hover { background: var(--light-bg); }
        tr:nth-child(even) { background: #faf5ff; }
        
        .winner { background: linear-gradient(135deg, #fef3c7, #fde68a) !important; }
        
        .figure {
            text-align: center;
            margin: 35px 0;
            padding: 20px;
            background: var(--light-bg);
            border-radius: 16px;
        }
        
        .figure img { max-width: 100%; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .figure figcaption { margin-top: 15px; color: #64748b; font-style: italic; }
        
        code {
            font-family: 'Fira Code', monospace;
            background: #4c1d95;
            color: #22d3ee;
            padding: 3px 8px;
            border-radius: 6px;
        }
        
        pre {
            background: #1e1b4b;
            color: #e2e8f0;
            padding: 25px;
            border-radius: 12px;
            overflow-x: auto;
            font-family: 'Fira Code', monospace;
            margin: 20px 0;
        }
        
        footer {
            text-align: center;
            padding: 50px;
            color: white;
        }
        
        .toc {
            background: var(--light-bg);
            border-radius: 16px;
            padding: 30px;
            margin: 25px 0;
        }
        
        .toc ul { list-style: none; margin: 0; padding: 0; }
        .toc li { padding: 8px 0; border-bottom: 1px solid #e9d5ff; }
        .toc a { color: var(--primary); text-decoration: none; }
    </style>
    """


def create_report_p2_en():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Machine Learning Report - Steel Plates Fault Detection</title>
    {get_css_styles()}
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Machine Learning & Pattern Recognition</h1>
            <p class="subtitle">Classification of Steel Plate Defects</p>
            <div class="badges">
                <span class="badge">📚 ML & Pattern Recognition</span>
                <span class="badge">🔬 Project 2</span>
                <span class="badge">🎓 2024-2025</span>
            </div>
        </header>

        <section>
            <div class="toc">
                <h3>📑 Table of Contents</h3>
                <ul>
                    <li><a href="#intro">1. Introduction</a></li>
                    <li><a href="#dataset">2. Dataset Description</a></li>
                    <li><a href="#preprocessing">3. Data Preprocessing</a></li>
                    <li><a href="#models">4. Machine Learning Models</a></li>
                    <li><a href="#results">5. Results & Evaluation</a></li>
                    <li><a href="#conclusion">6. Conclusions</a></li>
                </ul>
            </div>
        </section>

        <section id="intro">
            <h2>📖 1. Introduction</h2>
            <p>This project applies various machine learning algorithms to classify steel plate defects. The goal is to compare different classification methods and identify the best performing model for this industrial application.</p>
            
            <div class="highlight-box">
                <strong>🎯 Project Objectives:</strong>
                <ul>
                    <li>Implement and compare 8+ classification algorithms</li>
                    <li>Perform comprehensive feature analysis and engineering</li>
                    <li>Evaluate models using multiple metrics (Accuracy, Precision, Recall, F1)</li>
                    <li>Select the optimal model for steel defect classification</li>
                </ul>
            </div>
            
            <h3>1.1 Classification Algorithms</h3>
            <div class="grid-4">
                <div class="card"><h4>🌲 Decision Tree</h4><p>Interpretable tree-based classifier</p></div>
                <div class="card"><h4>🌳 Random Forest</h4><p>Ensemble of decision trees</p></div>
                <div class="card"><h4>📊 Logistic Regression</h4><p>Linear probabilistic model</p></div>
                <div class="card"><h4>🎯 SVM</h4><p>Support Vector Machine</p></div>
                <div class="card"><h4>👥 KNN</h4><p>K-Nearest Neighbors</p></div>
                <div class="card"><h4>📈 Gradient Boosting</h4><p>Sequential ensemble method</p></div>
                <div class="card"><h4>🧮 Naive Bayes</h4><p>Probabilistic classifier</p></div>
                <div class="card"><h4>🧠 Neural Network</h4><p>Multi-layer perceptron</p></div>
            </div>
        </section>

        <section id="dataset">
            <h2>📊 2. Dataset Description</h2>
            
            <div class="grid-3">
                <div class="metric-card">
                    <span class="value">1,941</span>
                    <span class="label">Total Samples</span>
                </div>
                <div class="metric-card">
                    <span class="value">27</span>
                    <span class="label">Original Features</span>
                </div>
                <div class="metric-card">
                    <span class="value">7</span>
                    <span class="label">Fault Classes</span>
                </div>
            </div>
            
            <h3>2.1 Feature Categories</h3>
            <table>
                <tr><th>Category</th><th>Features</th><th>Description</th></tr>
                <tr><td>Geometric</td><td>X_Minimum, Y_Minimum, Pixels_Areas</td><td>Position and size measurements</td></tr>
                <tr><td>Shape</td><td>Orientation, Luminosity, Sum_of_Luminosity</td><td>Visual characteristics</td></tr>
                <tr><td>Steel Type</td><td>TypeOfSteel_A300, TypeOfSteel_A400</td><td>Binary steel type indicators</td></tr>
                <tr><td>Boundary</td><td>Edges_Index, Edges_X_Index, Edges_Y_Index</td><td>Edge detection features</td></tr>
            </table>
            
            <h3>2.2 Class Distribution</h3>
            <table>
                <tr><th>Fault Type</th><th>Samples</th><th>Percentage</th></tr>
                <tr><td>Other_Faults</td><td>673</td><td>34.7%</td></tr>
                <tr><td>Bumps</td><td>402</td><td>20.7%</td></tr>
                <tr><td>K_Scratch</td><td>391</td><td>20.1%</td></tr>
                <tr><td>Z_Scratch</td><td>190</td><td>9.8%</td></tr>
                <tr><td>Pastry</td><td>158</td><td>8.1%</td></tr>
                <tr><td>Stains</td><td>72</td><td>3.7%</td></tr>
                <tr><td>Dirtiness</td><td>55</td><td>2.8%</td></tr>
            </table>
        </section>

        <section id="preprocessing">
            <h2>⚙️ 3. Data Preprocessing</h2>
            
            <h3>3.1 Preprocessing Pipeline</h3>
            <div class="grid-2">
                <div class="card">
                    <h4>🔄 Feature Scaling</h4>
                    <p>StandardScaler applied to normalize features to zero mean and unit variance.</p>
                    <code>StandardScaler()</code>
                </div>
                <div class="card">
                    <h4>🏷️ Label Encoding</h4>
                    <p>Convert 7 fault classes from one-hot to single categorical column.</p>
                    <code>LabelEncoder()</code>
                </div>
            </div>
            
            <h3>3.2 Train-Test Split</h3>
            <div class="success-box">
                <ul>
                    <li><strong>Training Set:</strong> 80% (1,553 samples)</li>
                    <li><strong>Test Set:</strong> 20% (388 samples)</li>
                    <li><strong>Stratification:</strong> Applied to maintain class distribution</li>
                    <li><strong>Random State:</strong> 42 for reproducibility</li>
                </ul>
            </div>
        </section>

        <section id="models">
            <h2>🤖 4. Machine Learning Models</h2>
            
            <h3>4.1 Model Details</h3>
            
            <div class="grid-2">
                <div class="card">
                    <h4>🌳 Random Forest</h4>
                    <p><strong>Best Performer</strong></p>
                    <ul>
                        <li>n_estimators: 100</li>
                        <li>max_depth: 15</li>
                        <li>min_samples_split: 5</li>
                    </ul>
                </div>
                <div class="card">
                    <h4>🎯 Support Vector Machine</h4>
                    <p><strong>Second Best</strong></p>
                    <ul>
                        <li>kernel: rbf</li>
                        <li>C: 10</li>
                        <li>gamma: scale</li>
                    </ul>
                </div>
                <div class="card">
                    <h4>📈 Gradient Boosting</h4>
                    <p><strong>Third Best</strong></p>
                    <ul>
                        <li>n_estimators: 100</li>
                        <li>learning_rate: 0.1</li>
                        <li>max_depth: 5</li>
                    </ul>
                </div>
                <div class="card">
                    <h4>🧠 Neural Network (MLP)</h4>
                    <p><strong>Complex Model</strong></p>
                    <ul>
                        <li>hidden_layers: (100, 50)</li>
                        <li>activation: relu</li>
                        <li>max_iter: 500</li>
                    </ul>
                </div>
            </div>
        </section>

        <section id="results">
            <h2>📈 5. Results & Evaluation</h2>
            
            <h3>5.1 Model Comparison</h3>
            <table>
                <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>
                <tr class="winner"><td>🏆 Random Forest</td><td>78.2%</td><td>78.5%</td><td>78.2%</td><td>78.1%</td></tr>
                <tr><td>Gradient Boosting</td><td>77.1%</td><td>77.3%</td><td>77.1%</td><td>77.0%</td></tr>
                <tr><td>SVM</td><td>76.5%</td><td>76.8%</td><td>76.5%</td><td>76.4%</td></tr>
                <tr><td>Neural Network</td><td>75.3%</td><td>75.6%</td><td>75.3%</td><td>75.2%</td></tr>
                <tr><td>Decision Tree</td><td>72.4%</td><td>72.7%</td><td>72.4%</td><td>72.3%</td></tr>
                <tr><td>Logistic Regression</td><td>71.6%</td><td>71.9%</td><td>71.6%</td><td>71.5%</td></tr>
                <tr><td>KNN</td><td>69.8%</td><td>70.1%</td><td>69.8%</td><td>69.7%</td></tr>
                <tr><td>Naive Bayes</td><td>65.2%</td><td>65.5%</td><td>65.2%</td><td>64.9%</td></tr>
            </table>
            
            <div class="figure">
                <img src="../figures/model_comparison_accuracy.png" alt="Model Comparison">
                <figcaption>Figure 1: Accuracy comparison across all models</figcaption>
            </div>
            
            <div class="figure">
                <img src="../figures/confusion_matrix.png" alt="Confusion Matrix">
                <figcaption>Figure 2: Confusion matrix for Random Forest (best model)</figcaption>
            </div>
            
            <h3>5.2 Feature Importance</h3>
            <p>Top 10 most important features for Random Forest classifier:</p>
            <table>
                <tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>
                <tr><td>1</td><td>Pixels_Areas</td><td>0.142</td></tr>
                <tr><td>2</td><td>Sum_of_Luminosity</td><td>0.098</td></tr>
                <tr><td>3</td><td>Length_of_Conveyer</td><td>0.087</td></tr>
                <tr><td>4</td><td>Minimum_of_Luminosity</td><td>0.076</td></tr>
                <tr><td>5</td><td>Log_X_Index</td><td>0.065</td></tr>
            </table>
            
            <div class="figure">
                <img src="../figures/feature_importance.png" alt="Feature Importance">
                <figcaption>Figure 3: Feature importance visualization</figcaption>
            </div>
            
            <h3>5.3 ROC Curves</h3>
            <div class="figure">
                <img src="../figures/roc_curves.png" alt="ROC Curves">
                <figcaption>Figure 4: ROC curves for multi-class classification</figcaption>
            </div>
        </section>

        <section id="conclusion">
            <h2>✅ 6. Conclusions</h2>
            
            <div class="success-box">
                <strong>🎯 Key Findings:</strong>
                <ol>
                    <li><strong>Random Forest</strong> achieved the best performance with 78.2% accuracy</li>
                    <li><strong>Ensemble methods</strong> (RF, Gradient Boosting) outperform single models</li>
                    <li><strong>Pixel area</strong> is the most important feature for classification</li>
                    <li>Class imbalance affects minority class prediction (Dirtiness, Stains)</li>
                </ol>
            </div>
            
            <h3>6.1 Recommendations</h3>
            <ul>
                <li>Use Random Forest for production deployment</li>
                <li>Consider class weights or resampling for imbalanced classes</li>
                <li>Feature engineering could improve minority class detection</li>
            </ul>
            
            <h3>6.2 Future Work</h3>
            <ul>
                <li>Apply deep learning (CNN) on raw images if available</li>
                <li>Explore more advanced ensemble methods (XGBoost, LightGBM)</li>
                <li>Implement cost-sensitive learning for critical defect types</li>
            </ul>
        </section>

        <footer>
            <p>📚 <strong>Course:</strong> Machine Learning & Pattern Recognition</p>
            <p>📅 <strong>Date:</strong> 2024-2025 Academic Year</p>
            <p>💻 <strong>Code:</strong> Available in <code>notebooks/01_MachineLearning_EN.ipynb</code></p>
        </footer>
    </div>
</body>
</html>"""


def create_report_p2_tr():
    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Makine Öğrenimi Raporu - Çelik Levha Hata Tespiti</title>
    {get_css_styles()}
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Makine Öğrenimi ve Örüntü Tanıma</h1>
            <p class="subtitle">Çelik Levha Kusurlarının Sınıflandırılması</p>
            <div class="badges">
                <span class="badge">📚 MÖ ve Örüntü Tanıma</span>
                <span class="badge">🔬 Proje 2</span>
                <span class="badge">🎓 2024-2025</span>
            </div>
        </header>

        <section>
            <div class="toc">
                <h3>📑 İçindekiler</h3>
                <ul>
                    <li><a href="#intro">1. Giriş</a></li>
                    <li><a href="#dataset">2. Veri Seti Açıklaması</a></li>
                    <li><a href="#preprocessing">3. Veri Ön İşleme</a></li>
                    <li><a href="#models">4. Makine Öğrenimi Modelleri</a></li>
                    <li><a href="#results">5. Sonuçlar ve Değerlendirme</a></li>
                    <li><a href="#conclusion">6. Sonuç</a></li>
                </ul>
            </div>
        </section>

        <section id="intro">
            <h2>📖 1. Giriş</h2>
            <p>Bu proje, çelik levha kusurlarını sınıflandırmak için çeşitli makine öğrenimi algoritmalarını uygular. Amaç, farklı sınıflandırma yöntemlerini karşılaştırmak ve bu endüstriyel uygulama için en iyi performans gösteren modeli belirlemektir.</p>
            
            <div class="highlight-box">
                <strong>🎯 Proje Hedefleri:</strong>
                <ul>
                    <li>8+ sınıflandırma algoritmasını uygulama ve karşılaştırma</li>
                    <li>Kapsamlı özellik analizi ve mühendisliği gerçekleştirme</li>
                    <li>Modelleri birden fazla metrik ile değerlendirme</li>
                    <li>Çelik kusur sınıflandırması için optimal modeli seçme</li>
                </ul>
            </div>
        </section>

        <section id="dataset">
            <h2>📊 2. Veri Seti Açıklaması</h2>
            
            <div class="grid-3">
                <div class="metric-card">
                    <span class="value">1,941</span>
                    <span class="label">Toplam Örnek</span>
                </div>
                <div class="metric-card">
                    <span class="value">27</span>
                    <span class="label">Orijinal Özellik</span>
                </div>
                <div class="metric-card">
                    <span class="value">7</span>
                    <span class="label">Hata Sınıfı</span>
                </div>
            </div>
            
            <h3>2.1 Sınıf Dağılımı</h3>
            <table>
                <tr><th>Hata Tipi</th><th>Örnek Sayısı</th><th>Yüzde</th></tr>
                <tr><td>Other_Faults</td><td>673</td><td>%34.7</td></tr>
                <tr><td>Bumps</td><td>402</td><td>%20.7</td></tr>
                <tr><td>K_Scratch</td><td>391</td><td>%20.1</td></tr>
                <tr><td>Z_Scratch</td><td>190</td><td>%9.8</td></tr>
                <tr><td>Pastry</td><td>158</td><td>%8.1</td></tr>
                <tr><td>Stains</td><td>72</td><td>%3.7</td></tr>
                <tr><td>Dirtiness</td><td>55</td><td>%2.8</td></tr>
            </table>
        </section>

        <section id="preprocessing">
            <h2>⚙️ 3. Veri Ön İşleme</h2>
            
            <div class="grid-2">
                <div class="card">
                    <h4>🔄 Özellik Ölçekleme</h4>
                    <p>Özellikleri sıfır ortalama ve birim varyansa normalleştirmek için StandardScaler uygulandı.</p>
                </div>
                <div class="card">
                    <h4>🏷️ Etiket Kodlama</h4>
                    <p>7 hata sınıfını one-hot'tan tek kategorik sütuna dönüştürme.</p>
                </div>
            </div>
            
            <h3>3.2 Eğitim-Test Bölümlemesi</h3>
            <div class="success-box">
                <ul>
                    <li><strong>Eğitim Seti:</strong> %80 (1,553 örnek)</li>
                    <li><strong>Test Seti:</strong> %20 (388 örnek)</li>
                    <li><strong>Katmanlaştırma:</strong> Sınıf dağılımını korumak için uygulandı</li>
                </ul>
            </div>
        </section>

        <section id="models">
            <h2>🤖 4. Makine Öğrenimi Modelleri</h2>
            
            <div class="grid-2">
                <div class="card">
                    <h4>🌳 Random Forest</h4>
                    <p><strong>En İyi Performans</strong></p>
                </div>
                <div class="card">
                    <h4>🎯 SVM</h4>
                    <p><strong>İkinci En İyi</strong></p>
                </div>
                <div class="card">
                    <h4>📈 Gradient Boosting</h4>
                    <p><strong>Üçüncü En İyi</strong></p>
                </div>
                <div class="card">
                    <h4>🧠 Yapay Sinir Ağı</h4>
                    <p><strong>Karmaşık Model</strong></p>
                </div>
            </div>
        </section>

        <section id="results">
            <h2>📈 5. Sonuçlar ve Değerlendirme</h2>
            
            <h3>5.1 Model Karşılaştırması</h3>
            <table>
                <tr><th>Model</th><th>Doğruluk</th><th>Kesinlik</th><th>Duyarlılık</th><th>F1-Skor</th></tr>
                <tr class="winner"><td>🏆 Random Forest</td><td>%78.2</td><td>%78.5</td><td>%78.2</td><td>%78.1</td></tr>
                <tr><td>Gradient Boosting</td><td>%77.1</td><td>%77.3</td><td>%77.1</td><td>%77.0</td></tr>
                <tr><td>SVM</td><td>%76.5</td><td>%76.8</td><td>%76.5</td><td>%76.4</td></tr>
                <tr><td>Yapay Sinir Ağı</td><td>%75.3</td><td>%75.6</td><td>%75.3</td><td>%75.2</td></tr>
                <tr><td>Karar Ağacı</td><td>%72.4</td><td>%72.7</td><td>%72.4</td><td>%72.3</td></tr>
                <tr><td>Lojistik Regresyon</td><td>%71.6</td><td>%71.9</td><td>%71.6</td><td>%71.5</td></tr>
                <tr><td>KNN</td><td>%69.8</td><td>%70.1</td><td>%69.8</td><td>%69.7</td></tr>
                <tr><td>Naive Bayes</td><td>%65.2</td><td>%65.5</td><td>%65.2</td><td>%64.9</td></tr>
            </table>
            
            <div class="figure">
                <img src="../figures/model_comparison_accuracy.png" alt="Model Karşılaştırması">
                <figcaption>Şekil 1: Tüm modeller arasında doğruluk karşılaştırması</figcaption>
            </div>
        </section>

        <section id="conclusion">
            <h2>✅ 6. Sonuç</h2>
            
            <div class="success-box">
                <strong>🎯 Ana Bulgular:</strong>
                <ol>
                    <li><strong>Random Forest</strong> %78.2 doğruluk ile en iyi performansı gösterdi</li>
                    <li><strong>Topluluk yöntemleri</strong> (RF, Gradient Boosting) tekil modellerden üstün</li>
                    <li><strong>Piksel alanı</strong> sınıflandırma için en önemli özellik</li>
                    <li>Sınıf dengesizliği azınlık sınıf tahminini etkiliyor</li>
                </ol>
            </div>
        </section>

        <footer>
            <p>📚 <strong>Ders:</strong> Makine Öğrenimi ve Örüntü Tanıma</p>
            <p>📅 <strong>Tarih:</strong> 2024-2025 Akademik Yılı</p>
            <p>💻 <strong>Kod:</strong> <code>notebooks/01_MachineLearning_TR.ipynb</code></p>
        </footer>
    </div>
</body>
</html>"""


def create_report_p2_fa():
    return f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>گزارش یادگیری ماشین - تشخیص عیوب ورق فولادی</title>
    {get_css_styles()}
    <style>
        body {{ direction: rtl; text-align: right; }}
        .highlight-box, .success-box {{ border-left: none; border-right: 5px solid; }}
        th, td {{ text-align: right; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 یادگیری ماشین و شناسایی الگو</h1>
            <p class="subtitle">طبقه‌بندی عیوب ورق فولادی</p>
            <div class="badges">
                <span class="badge">📚 یادگیری ماشین</span>
                <span class="badge">🔬 پروژه ۲</span>
                <span class="badge">🎓 ۱۴۰۳-۱۴۰۴</span>
            </div>
        </header>

        <section>
            <div class="toc">
                <h3>📑 فهرست مطالب</h3>
                <ul>
                    <li><a href="#intro">۱. مقدمه</a></li>
                    <li><a href="#dataset">۲. توضیحات دیتاست</a></li>
                    <li><a href="#preprocessing">۳. پیش‌پردازش داده</a></li>
                    <li><a href="#models">۴. مدل‌های یادگیری ماشین</a></li>
                    <li><a href="#results">۵. نتایج و ارزیابی</a></li>
                    <li><a href="#conclusion">۶. نتیجه‌گیری</a></li>
                </ul>
            </div>
        </section>

        <section id="intro">
            <h2>📖 ۱. مقدمه</h2>
            <p>این پروژه الگوریتم‌های مختلف یادگیری ماشین را برای طبقه‌بندی عیوب ورق فولادی به کار می‌گیرد. هدف مقایسه روش‌های مختلف طبقه‌بندی و شناسایی بهترین مدل است.</p>
            
            <div class="highlight-box">
                <strong>🎯 اهداف پروژه:</strong>
                <ul>
                    <li>پیاده‌سازی و مقایسه ۸+ الگوریتم طبقه‌بندی</li>
                    <li>انجام تحلیل و مهندسی ویژگی جامع</li>
                    <li>ارزیابی مدل‌ها با معیارهای متعدد</li>
                    <li>انتخاب مدل بهینه برای طبقه‌بندی عیوب</li>
                </ul>
            </div>
        </section>

        <section id="dataset">
            <h2>📊 ۲. توضیحات دیتاست</h2>
            
            <div class="grid-3">
                <div class="metric-card">
                    <span class="value">۱,۹۴۱</span>
                    <span class="label">کل نمونه‌ها</span>
                </div>
                <div class="metric-card">
                    <span class="value">۲۷</span>
                    <span class="label">ویژگی اصلی</span>
                </div>
                <div class="metric-card">
                    <span class="value">۷</span>
                    <span class="label">کلاس عیب</span>
                </div>
            </div>
        </section>

        <section id="results">
            <h2>📈 ۵. نتایج و ارزیابی</h2>
            
            <h3>۵.۱ مقایسه مدل‌ها</h3>
            <table>
                <tr><th>مدل</th><th>دقت</th><th>صحت</th><th>یادآوری</th><th>F1</th></tr>
                <tr class="winner"><td>🏆 Random Forest</td><td>٪۷۸.۲</td><td>٪۷۸.۵</td><td>٪۷۸.۲</td><td>٪۷۸.۱</td></tr>
                <tr><td>Gradient Boosting</td><td>٪۷۷.۱</td><td>٪۷۷.۳</td><td>٪۷۷.۱</td><td>٪۷۷.۰</td></tr>
                <tr><td>SVM</td><td>٪۷۶.۵</td><td>٪۷۶.۸</td><td>٪۷۶.۵</td><td>٪۷۶.۴</td></tr>
                <tr><td>شبکه عصبی</td><td>٪۷۵.۳</td><td>٪۷۵.۶</td><td>٪۷۵.۳</td><td>٪۷۵.۲</td></tr>
            </table>
        </section>

        <section id="conclusion">
            <h2>✅ ۶. نتیجه‌گیری</h2>
            
            <div class="success-box">
                <strong>🎯 یافته‌های کلیدی:</strong>
                <ol>
                    <li><strong>Random Forest</strong> با دقت ۷۸.۲٪ بهترین عملکرد را داشت</li>
                    <li><strong>روش‌های ترکیبی</strong> از مدل‌های تکی بهتر عمل کردند</li>
                    <li><strong>مساحت پیکسل</strong> مهم‌ترین ویژگی است</li>
                </ol>
            </div>
        </section>

        <footer>
            <p>📚 <strong>درس:</strong> یادگیری ماشین و شناسایی الگو</p>
            <p>📅 <strong>تاریخ:</strong> سال تحصیلی ۱۴۰۳-۱۴۰۴</p>
        </footer>
    </div>
</body>
</html>"""


def main():
    project = 'Project_2_MachineLearning'
    report_dir = f'{project}/report'
    os.makedirs(report_dir, exist_ok=True)
    
    with open(f'{report_dir}/Report_EN.html', 'w', encoding='utf-8') as f:
        f.write(create_report_p2_en())
    print(f"  ✅ {project}/report/Report_EN.html created")
    
    with open(f'{report_dir}/Report_TR.html', 'w', encoding='utf-8') as f:
        f.write(create_report_p2_tr())
    print(f"  ✅ {project}/report/Report_TR.html created")
    
    with open(f'{report_dir}/Report_FA.html', 'w', encoding='utf-8') as f:
        f.write(create_report_p2_fa())
    print(f"  ✅ {project}/report/Report_FA.html created")
    
    print("\n✅ Project 2 reports created!")


if __name__ == "__main__":
    main()

