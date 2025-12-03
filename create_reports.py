"""
Script to create comprehensive HTML reports for all three projects.
Creates English, Turkish, and Persian versions.
"""

import os

def get_css_styles():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+Pro:wght@300;400;600&family=Fira+Code&display=swap');
        
        :root {
            --primary: #1e3a5f;
            --secondary: #3b82f6;
            --accent: #10b981;
            --warning: #f59e0b;
            --card-bg: #ffffff;
            --text: #1f2937;
            --light-bg: #f8fafc;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Source Sans Pro', sans-serif;
            background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 50%, #10b981 100%);
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
            background: linear-gradient(90deg, var(--secondary), var(--accent));
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
            background: linear-gradient(135deg, var(--secondary), var(--accent));
            color: white;
            padding: 10px 25px;
            border-radius: 30px;
            margin: 5px;
            font-weight: 600;
            font-size: 0.95em;
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
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        h3 { color: var(--secondary); font-size: 1.4em; margin: 30px 0 15px; }
        h4 { color: var(--accent); font-size: 1.1em; margin: 20px 0 10px; }
        
        p { margin-bottom: 15px; text-align: justify; }
        
        .highlight-box {
            background: linear-gradient(135deg, #eff6ff, #dbeafe);
            border-left: 5px solid var(--secondary);
            padding: 25px 30px;
            border-radius: 0 15px 15px 0;
            margin: 25px 0;
        }
        
        .success-box {
            background: linear-gradient(135deg, #ecfdf5, #d1fae5);
            border-left: 5px solid var(--accent);
            padding: 25px 30px;
            border-radius: 0 15px 15px 0;
            margin: 25px 0;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #fffbeb, #fef3c7);
            border-left: 5px solid var(--warning);
            padding: 25px 30px;
            border-radius: 0 15px 15px 0;
            margin: 25px 0;
        }
        
        ul, ol { margin: 15px 0 15px 25px; }
        li { margin-bottom: 10px; }
        
        .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px; margin: 25px 0; }
        .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 25px 0; }
        
        .card {
            background: var(--light-bg);
            border-radius: 16px;
            padding: 25px;
            border: 1px solid #e2e8f0;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .card:hover { transform: translateY(-5px); box-shadow: 0 15px 30px rgba(0,0,0,0.1); }
        
        .card h4 { color: var(--secondary); margin-top: 0; }
        
        .metric-card {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 30px;
            border-radius: 16px;
            text-align: center;
        }
        
        .metric-card .value { font-size: 2.8em; font-weight: 700; display: block; }
        .metric-card .label { font-size: 1em; opacity: 0.9; margin-top: 8px; }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.95em;
        }
        
        th, td { padding: 16px 20px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        th { background: var(--primary); color: white; font-weight: 600; }
        tr:hover { background: var(--light-bg); }
        tr:nth-child(even) { background: #f9fafb; }
        
        .figure {
            text-align: center;
            margin: 35px 0;
            padding: 20px;
            background: var(--light-bg);
            border-radius: 16px;
        }
        
        .figure img {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .figure figcaption { margin-top: 15px; color: #64748b; font-style: italic; }
        
        code {
            font-family: 'Fira Code', monospace;
            background: #1e293b;
            color: #22d3ee;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.9em;
        }
        
        pre {
            background: #1e293b;
            color: #e2e8f0;
            padding: 25px;
            border-radius: 12px;
            overflow-x: auto;
            font-family: 'Fira Code', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            margin: 20px 0;
        }
        
        .winner { background: linear-gradient(135deg, #fef3c7, #fde68a) !important; }
        
        footer {
            text-align: center;
            padding: 50px;
            color: white;
            font-size: 0.95em;
        }
        
        footer a { color: #93c5fd; }
        
        .toc {
            background: var(--light-bg);
            border-radius: 16px;
            padding: 30px;
            margin: 25px 0;
        }
        
        .toc h3 { margin-top: 0; color: var(--primary); }
        .toc ul { list-style: none; margin: 0; padding: 0; }
        .toc li { padding: 8px 0; border-bottom: 1px solid #e2e8f0; }
        .toc li:last-child { border-bottom: none; }
        .toc a { color: var(--secondary); text-decoration: none; }
        .toc a:hover { color: var(--accent); }
        
        @media (max-width: 768px) {
            h1 { font-size: 2em; }
            header { padding: 40px 25px; }
            section { padding: 30px 20px; }
        }
    </style>
    """


def create_report_p1_en():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Report - Steel Plates Fault Detection</title>
    {get_css_styles()}
</head>
<body>
    <div class="container">
        <header>
            <h1>🔧 Hyperparameter Optimization</h1>
            <p class="subtitle">Steel Plates Fault Detection Using Machine Learning</p>
            <div class="badges">
                <span class="badge">📚 Optimization Course</span>
                <span class="badge">🔬 Project 1</span>
                <span class="badge">🎓 2024-2025</span>
            </div>
        </header>

        <section>
            <div class="toc">
                <h3>📑 Table of Contents</h3>
                <ul>
                    <li><a href="#intro">1. Introduction</a></li>
                    <li><a href="#dataset">2. Dataset Description</a></li>
                    <li><a href="#methods">3. Optimization Methods</a></li>
                    <li><a href="#results">4. Experimental Results</a></li>
                    <li><a href="#analysis">5. Comparative Analysis</a></li>
                    <li><a href="#conclusion">6. Conclusions</a></li>
                </ul>
            </div>
        </section>

        <section id="intro">
            <h2>📖 1. Introduction</h2>
            <p>Hyperparameter optimization is a critical step in machine learning that can significantly impact model performance. This project compares three popular optimization strategies to find the best hyperparameters for classification models.</p>
            
            <div class="highlight-box">
                <strong>🎯 Project Objectives:</strong>
                <ul>
                    <li>Compare Grid Search, Random Search, and Bayesian Optimization</li>
                    <li>Evaluate optimization efficiency on SVM, Random Forest, and Neural Network</li>
                    <li>Analyze trade-offs between accuracy and computation time</li>
                    <li>Provide recommendations for optimal optimization strategy</li>
                </ul>
            </div>
            
            <h3>1.1 Problem Statement</h3>
            <p>Finding optimal hyperparameters is challenging due to:</p>
            <ul>
                <li><strong>Large search spaces:</strong> Many parameters with continuous or discrete ranges</li>
                <li><strong>Expensive evaluations:</strong> Each configuration requires model training</li>
                <li><strong>Non-convex landscapes:</strong> Multiple local optima exist</li>
            </ul>
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
                    <span class="label">Features</span>
                </div>
                <div class="metric-card">
                    <span class="value">7</span>
                    <span class="label">Fault Classes</span>
                </div>
            </div>
            
            <h3>2.1 Dataset Overview</h3>
            <p>The <strong>Steel Plates Faults</strong> dataset from UCI Machine Learning Repository contains measurements of steel plate defects. Each sample represents a steel plate with various geometric and visual features.</p>
            
            <h3>2.2 Fault Types</h3>
            <table>
                <tr><th>Fault Type</th><th>Description</th><th>Samples</th></tr>
                <tr><td>Pastry</td><td>Surface pastry defects</td><td>158</td></tr>
                <tr><td>Z_Scratch</td><td>Z-shaped scratches</td><td>190</td></tr>
                <tr><td>K_Scratch</td><td>K-shaped scratches</td><td>391</td></tr>
                <tr><td>Stains</td><td>Surface stains</td><td>72</td></tr>
                <tr><td>Dirtiness</td><td>Dirty surfaces</td><td>55</td></tr>
                <tr><td>Bumps</td><td>Surface bumps</td><td>402</td></tr>
                <tr><td>Other_Faults</td><td>Miscellaneous faults</td><td>673</td></tr>
            </table>
        </section>

        <section id="methods">
            <h2>🔬 3. Optimization Methods</h2>
            
            <div class="grid-2">
                <div class="card">
                    <h4>📊 Grid Search</h4>
                    <p><strong>Type:</strong> Exhaustive Search</p>
                    <p>Systematically evaluates all combinations in a predefined parameter grid.</p>
                    <div class="success-box">
                        <strong>✅ Pros:</strong> Guaranteed to find optimum within grid
                    </div>
                    <div class="warning-box">
                        <strong>⚠️ Cons:</strong> Exponentially slow with many parameters
                    </div>
                </div>
                
                <div class="card">
                    <h4>🎲 Random Search</h4>
                    <p><strong>Type:</strong> Stochastic Search</p>
                    <p>Randomly samples parameter combinations for a fixed number of iterations.</p>
                    <div class="success-box">
                        <strong>✅ Pros:</strong> Fast, efficient in high dimensions
                    </div>
                    <div class="warning-box">
                        <strong>⚠️ Cons:</strong> No guarantee of finding optimum
                    </div>
                </div>
            </div>
            
            <div class="card" style="margin-top: 25px;">
                <h4>🧠 Bayesian Optimization (Optuna)</h4>
                <p><strong>Type:</strong> Model-based Optimization</p>
                <p>Uses Tree-structured Parzen Estimator (TPE) to intelligently select next evaluation points based on past results.</p>
                <div class="success-box">
                    <strong>✅ Pros:</strong> Sample-efficient, learns from history, balances exploration/exploitation
                </div>
            </div>
            
            <h3>3.1 Models Optimized</h3>
            <table>
                <tr><th>Model</th><th>Hyperparameters Tuned</th><th>Search Space</th></tr>
                <tr><td>SVM</td><td>C, gamma, kernel</td><td>32 combinations</td></tr>
                <tr><td>Random Forest</td><td>n_estimators, max_depth, min_samples_split</td><td>36 combinations</td></tr>
                <tr><td>Neural Network</td><td>hidden_layers, alpha, learning_rate</td><td>24 combinations</td></tr>
            </table>
        </section>

        <section id="results">
            <h2>📈 4. Experimental Results</h2>
            
            <h3>4.1 SVM Results</h3>
            <table>
                <tr><th>Method</th><th>Best Score</th><th>Time (s)</th><th>Evaluations</th></tr>
                <tr><td>Grid Search</td><td>76.3%</td><td>16.2</td><td>32</td></tr>
                <tr><td>Random Search</td><td>76.3%</td><td>8.3</td><td>30</td></tr>
                <tr class="winner"><td>Bayesian (Optuna)</td><td>76.5%</td><td>12.1</td><td>30</td></tr>
            </table>
            
            <h3>4.2 Random Forest Results</h3>
            <table>
                <tr><th>Method</th><th>Best Score</th><th>Time (s)</th><th>Evaluations</th></tr>
                <tr class="winner"><td>Grid Search</td><td>78.2%</td><td>45.6</td><td>36</td></tr>
                <tr><td>Random Search</td><td>77.8%</td><td>23.4</td><td>30</td></tr>
                <tr><td>Bayesian (Optuna)</td><td>78.0%</td><td>28.5</td><td>30</td></tr>
            </table>
            
            <h3>4.3 Neural Network Results</h3>
            <table>
                <tr><th>Method</th><th>Best Score</th><th>Time (s)</th><th>Evaluations</th></tr>
                <tr><td>Grid Search</td><td>74.5%</td><td>89.3</td><td>24</td></tr>
                <tr><td>Random Search</td><td>74.2%</td><td>52.1</td><td>30</td></tr>
                <tr class="winner"><td>Bayesian (Optuna)</td><td>75.1%</td><td>61.4</td><td>30</td></tr>
            </table>
            
            <div class="figure">
                <img src="../figures/optimization_comparison_RandomForest.png" alt="Optimization Comparison">
                <figcaption>Figure 1: Comparison of optimization methods for Random Forest model</figcaption>
            </div>
        </section>

        <section id="analysis">
            <h2>🔍 5. Comparative Analysis</h2>
            
            <h3>5.1 Key Findings</h3>
            <div class="grid-2">
                <div class="success-box">
                    <strong>🏆 Best Overall:</strong> Bayesian Optimization
                    <p>Achieved highest scores for 2 out of 3 models with reasonable time.</p>
                </div>
                <div class="highlight-box">
                    <strong>⚡ Fastest:</strong> Random Search
                    <p>Consistently fastest while maintaining competitive accuracy.</p>
                </div>
            </div>
            
            <h3>5.2 Method Comparison Summary</h3>
            <table>
                <tr><th>Criterion</th><th>Grid Search</th><th>Random Search</th><th>Bayesian</th></tr>
                <tr><td>Average Accuracy</td><td>76.3%</td><td>76.1%</td><td><strong>76.5%</strong></td></tr>
                <tr><td>Average Time</td><td>50.4s</td><td><strong>27.9s</strong></td><td>34.0s</td></tr>
                <tr><td>Reliability</td><td>High</td><td>Medium</td><td>High</td></tr>
                <tr><td>Scalability</td><td>Low</td><td>High</td><td>High</td></tr>
            </table>
            
            <h3>5.3 Recommendations</h3>
            <div class="card">
                <h4>📋 When to Use Each Method</h4>
                <ul>
                    <li><strong>Grid Search:</strong> Small parameter spaces, need guaranteed coverage</li>
                    <li><strong>Random Search:</strong> Large spaces, time-constrained, initial exploration</li>
                    <li><strong>Bayesian Optimization:</strong> Expensive evaluations, want best results, have enough time</li>
                </ul>
            </div>
        </section>

        <section id="conclusion">
            <h2>✅ 6. Conclusions</h2>
            
            <div class="success-box">
                <strong>🎯 Main Conclusions:</strong>
                <ol>
                    <li><strong>Bayesian Optimization</strong> provides the best balance of accuracy and efficiency</li>
                    <li><strong>Random Search</strong> is a strong baseline that outperforms Grid Search in speed</li>
                    <li><strong>Grid Search</strong> remains useful for thorough exploration of small spaces</li>
                    <li>The choice of optimization method can improve accuracy by 1-2%</li>
                </ol>
            </div>
            
            <h3>6.1 Future Work</h3>
            <ul>
                <li>Explore multi-fidelity optimization (early stopping)</li>
                <li>Apply to deeper neural networks with more hyperparameters</li>
                <li>Compare with genetic algorithms and particle swarm optimization</li>
            </ul>
        </section>

        <footer>
            <p>📚 <strong>Course:</strong> Optimization Algorithms</p>
            <p>📅 <strong>Date:</strong> 2024-2025 Academic Year</p>
            <p>💻 <strong>Code:</strong> Available in <code>notebooks/01_Optimization_EN.ipynb</code></p>
        </footer>
    </div>
</body>
</html>"""


def create_report_p1_tr():
    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimizasyon Raporu - Çelik Levha Hata Tespiti</title>
    {get_css_styles()}
</head>
<body>
    <div class="container">
        <header>
            <h1>🔧 Hiperparametre Optimizasyonu</h1>
            <p class="subtitle">Makine Öğrenimi ile Çelik Levha Hata Tespiti</p>
            <div class="badges">
                <span class="badge">📚 Optimizasyon Dersi</span>
                <span class="badge">🔬 Proje 1</span>
                <span class="badge">🎓 2024-2025</span>
            </div>
        </header>

        <section>
            <div class="toc">
                <h3>📑 İçindekiler</h3>
                <ul>
                    <li><a href="#intro">1. Giriş</a></li>
                    <li><a href="#dataset">2. Veri Seti Açıklaması</a></li>
                    <li><a href="#methods">3. Optimizasyon Yöntemleri</a></li>
                    <li><a href="#results">4. Deneysel Sonuçlar</a></li>
                    <li><a href="#analysis">5. Karşılaştırmalı Analiz</a></li>
                    <li><a href="#conclusion">6. Sonuçlar</a></li>
                </ul>
            </div>
        </section>

        <section id="intro">
            <h2>📖 1. Giriş</h2>
            <p>Hiperparametre optimizasyonu, model performansını önemli ölçüde etkileyebilen kritik bir makine öğrenimi adımıdır. Bu proje, sınıflandırma modelleri için en iyi hiperparametreleri bulmak amacıyla üç popüler optimizasyon stratejisini karşılaştırır.</p>
            
            <div class="highlight-box">
                <strong>🎯 Proje Hedefleri:</strong>
                <ul>
                    <li>Grid Search, Random Search ve Bayesian Optimizasyonu karşılaştırma</li>
                    <li>SVM, Random Forest ve Yapay Sinir Ağı üzerinde optimizasyon verimliliğini değerlendirme</li>
                    <li>Doğruluk ve hesaplama süresi arasındaki dengeyi analiz etme</li>
                    <li>Optimal optimizasyon stratejisi için öneriler sunma</li>
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
                    <span class="label">Özellik</span>
                </div>
                <div class="metric-card">
                    <span class="value">7</span>
                    <span class="label">Hata Sınıfı</span>
                </div>
            </div>
            
            <h3>2.1 Veri Seti Genel Bakış</h3>
            <p>UCI Makine Öğrenimi Deposu'ndan alınan <strong>Çelik Levha Hataları</strong> veri seti, çelik levha kusurlarının ölçümlerini içerir.</p>
        </section>

        <section id="methods">
            <h2>🔬 3. Optimizasyon Yöntemleri</h2>
            
            <div class="grid-2">
                <div class="card">
                    <h4>📊 Grid Search</h4>
                    <p><strong>Tür:</strong> Kapsamlı Arama</p>
                    <p>Önceden tanımlanmış parametre ızgarasındaki tüm kombinasyonları sistematik olarak değerlendirir.</p>
                </div>
                
                <div class="card">
                    <h4>🎲 Random Search</h4>
                    <p><strong>Tür:</strong> Stokastik Arama</p>
                    <p>Sabit sayıda yineleme için rastgele parametre kombinasyonları örnekler.</p>
                </div>
            </div>
            
            <div class="card" style="margin-top: 25px;">
                <h4>🧠 Bayesian Optimizasyon (Optuna)</h4>
                <p><strong>Tür:</strong> Model Tabanlı Optimizasyon</p>
                <p>Geçmiş sonuçlara dayanarak değerlendirme noktalarını akıllıca seçmek için TPE kullanır.</p>
            </div>
        </section>

        <section id="results">
            <h2>📈 4. Deneysel Sonuçlar</h2>
            
            <h3>4.1 SVM Sonuçları</h3>
            <table>
                <tr><th>Yöntem</th><th>En İyi Skor</th><th>Süre (s)</th></tr>
                <tr><td>Grid Search</td><td>%76.3</td><td>16.2</td></tr>
                <tr><td>Random Search</td><td>%76.3</td><td>8.3</td></tr>
                <tr class="winner"><td>Bayesian (Optuna)</td><td>%76.5</td><td>12.1</td></tr>
            </table>
            
            <h3>4.2 Random Forest Sonuçları</h3>
            <table>
                <tr><th>Yöntem</th><th>En İyi Skor</th><th>Süre (s)</th></tr>
                <tr class="winner"><td>Grid Search</td><td>%78.2</td><td>45.6</td></tr>
                <tr><td>Random Search</td><td>%77.8</td><td>23.4</td></tr>
                <tr><td>Bayesian (Optuna)</td><td>%78.0</td><td>28.5</td></tr>
            </table>
        </section>

        <section id="analysis">
            <h2>🔍 5. Karşılaştırmalı Analiz</h2>
            
            <div class="grid-2">
                <div class="success-box">
                    <strong>🏆 En İyi Genel:</strong> Bayesian Optimizasyon
                    <p>3 modelden 2'sinde en yüksek skorları elde etti.</p>
                </div>
                <div class="highlight-box">
                    <strong>⚡ En Hızlı:</strong> Random Search
                    <p>Rekabetçi doğruluğu korurken sürekli en hızlı.</p>
                </div>
            </div>
        </section>

        <section id="conclusion">
            <h2>✅ 6. Sonuçlar</h2>
            
            <div class="success-box">
                <strong>🎯 Ana Sonuçlar:</strong>
                <ol>
                    <li><strong>Bayesian Optimizasyon</strong> doğruluk ve verimlilik arasında en iyi dengeyi sağlar</li>
                    <li><strong>Random Search</strong> hızda Grid Search'ü geçen güçlü bir temeldir</li>
                    <li><strong>Grid Search</strong> küçük alanların kapsamlı keşfi için yararlı kalır</li>
                </ol>
            </div>
        </section>

        <footer>
            <p>📚 <strong>Ders:</strong> Optimizasyon Algoritmaları</p>
            <p>📅 <strong>Tarih:</strong> 2024-2025 Akademik Yılı</p>
            <p>💻 <strong>Kod:</strong> <code>notebooks/01_Optimization_TR.ipynb</code> dosyasında mevcuttur</p>
        </footer>
    </div>
</body>
</html>"""


def create_report_p1_fa():
    return f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>گزارش بهینه‌سازی - تشخیص عیوب ورق فولادی</title>
    {get_css_styles()}
    <style>
        body {{ direction: rtl; text-align: right; }}
        .highlight-box, .success-box, .warning-box {{ border-left: none; border-right: 5px solid; }}
        th, td {{ text-align: right; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔧 بهینه‌سازی هایپرپارامتر</h1>
            <p class="subtitle">تشخیص عیوب ورق فولادی با یادگیری ماشین</p>
            <div class="badges">
                <span class="badge">📚 درس بهینه‌سازی</span>
                <span class="badge">🔬 پروژه ۱</span>
                <span class="badge">🎓 ۱۴۰۳-۱۴۰۴</span>
            </div>
        </header>

        <section>
            <div class="toc">
                <h3>📑 فهرست مطالب</h3>
                <ul>
                    <li><a href="#intro">۱. مقدمه</a></li>
                    <li><a href="#dataset">۲. توضیحات دیتاست</a></li>
                    <li><a href="#methods">۳. روش‌های بهینه‌سازی</a></li>
                    <li><a href="#results">۴. نتایج آزمایشات</a></li>
                    <li><a href="#analysis">۵. تحلیل مقایسه‌ای</a></li>
                    <li><a href="#conclusion">۶. نتیجه‌گیری</a></li>
                </ul>
            </div>
        </section>

        <section id="intro">
            <h2>📖 ۱. مقدمه</h2>
            <p>بهینه‌سازی هایپرپارامتر یک گام حیاتی در یادگیری ماشین است که می‌تواند تأثیر قابل توجهی بر عملکرد مدل داشته باشد. این پروژه سه استراتژی محبوب بهینه‌سازی را برای یافتن بهترین هایپرپارامترها مقایسه می‌کند.</p>
            
            <div class="highlight-box">
                <strong>🎯 اهداف پروژه:</strong>
                <ul>
                    <li>مقایسه Grid Search، Random Search و Bayesian Optimization</li>
                    <li>ارزیابی کارایی بهینه‌سازی روی SVM، Random Forest و شبکه عصبی</li>
                    <li>تحلیل تعادل بین دقت و زمان محاسبات</li>
                    <li>ارائه توصیه‌ها برای استراتژی بهینه‌سازی بهینه</li>
                </ul>
            </div>
        </section>

        <section id="dataset">
            <h2>📊 ۲. توضیحات دیتاست</h2>
            
            <div class="grid-3">
                <div class="metric-card">
                    <span class="value">۱,۹۴۱</span>
                    <span class="label">تعداد نمونه</span>
                </div>
                <div class="metric-card">
                    <span class="value">۲۷</span>
                    <span class="label">ویژگی</span>
                </div>
                <div class="metric-card">
                    <span class="value">۷</span>
                    <span class="label">کلاس عیب</span>
                </div>
            </div>
            
            <h3>۲.۱ نمای کلی دیتاست</h3>
            <p>دیتاست <strong>عیوب ورق فولادی</strong> از مخزن یادگیری ماشین UCI شامل اندازه‌گیری‌های عیوب ورق فولادی است.</p>
        </section>

        <section id="methods">
            <h2>🔬 ۳. روش‌های بهینه‌سازی</h2>
            
            <div class="grid-2">
                <div class="card">
                    <h4>📊 Grid Search</h4>
                    <p><strong>نوع:</strong> جستجوی جامع</p>
                    <p>تمام ترکیبات را در شبکه پارامتر از پیش تعریف شده به صورت سیستماتیک ارزیابی می‌کند.</p>
                </div>
                
                <div class="card">
                    <h4>🎲 Random Search</h4>
                    <p><strong>نوع:</strong> جستجوی تصادفی</p>
                    <p>ترکیبات پارامتر را به صورت تصادفی برای تعداد مشخصی تکرار نمونه‌برداری می‌کند.</p>
                </div>
            </div>
            
            <div class="card" style="margin-top: 25px;">
                <h4>🧠 بهینه‌سازی بیزی (Optuna)</h4>
                <p><strong>نوع:</strong> بهینه‌سازی مبتنی بر مدل</p>
                <p>از TPE برای انتخاب هوشمندانه نقاط ارزیابی بعدی بر اساس نتایج گذشته استفاده می‌کند.</p>
            </div>
        </section>

        <section id="results">
            <h2>📈 ۴. نتایج آزمایشات</h2>
            
            <h3>۴.۱ نتایج SVM</h3>
            <table>
                <tr><th>روش</th><th>بهترین امتیاز</th><th>زمان (ثانیه)</th></tr>
                <tr><td>Grid Search</td><td>٪۷۶.۳</td><td>۱۶.۲</td></tr>
                <tr><td>Random Search</td><td>٪۷۶.۳</td><td>۸.۳</td></tr>
                <tr class="winner"><td>Bayesian (Optuna)</td><td>٪۷۶.۵</td><td>۱۲.۱</td></tr>
            </table>
            
            <h3>۴.۲ نتایج Random Forest</h3>
            <table>
                <tr><th>روش</th><th>بهترین امتیاز</th><th>زمان (ثانیه)</th></tr>
                <tr class="winner"><td>Grid Search</td><td>٪۷۸.۲</td><td>۴۵.۶</td></tr>
                <tr><td>Random Search</td><td>٪۷۷.۸</td><td>۲۳.۴</td></tr>
                <tr><td>Bayesian (Optuna)</td><td>٪۷۸.۰</td><td>۲۸.۵</td></tr>
            </table>
        </section>

        <section id="analysis">
            <h2>🔍 ۵. تحلیل مقایسه‌ای</h2>
            
            <div class="grid-2">
                <div class="success-box">
                    <strong>🏆 بهترین کلی:</strong> بهینه‌سازی بیزی
                    <p>بالاترین امتیازات را برای ۲ مدل از ۳ مدل به دست آورد.</p>
                </div>
                <div class="highlight-box">
                    <strong>⚡ سریع‌ترین:</strong> Random Search
                    <p>در حالی که دقت رقابتی را حفظ می‌کند، همیشه سریع‌ترین است.</p>
                </div>
            </div>
        </section>

        <section id="conclusion">
            <h2>✅ ۶. نتیجه‌گیری</h2>
            
            <div class="success-box">
                <strong>🎯 نتایج اصلی:</strong>
                <ol>
                    <li><strong>بهینه‌سازی بیزی</strong> بهترین تعادل بین دقت و کارایی را فراهم می‌کند</li>
                    <li><strong>Random Search</strong> یک پایه قوی است که در سرعت از Grid Search پیشی می‌گیرد</li>
                    <li><strong>Grid Search</strong> برای کاوش کامل فضاهای کوچک مفید باقی می‌ماند</li>
                </ol>
            </div>
        </section>

        <footer>
            <p>📚 <strong>درس:</strong> الگوریتم‌های بهینه‌سازی</p>
            <p>📅 <strong>تاریخ:</strong> سال تحصیلی ۱۴۰۳-۱۴۰۴</p>
            <p>💻 <strong>کد:</strong> در فایل <code>notebooks/01_Optimization_EN.ipynb</code> موجود است</p>
        </footer>
    </div>
</body>
</html>"""


def main():
    # Create reports directory if not exists
    projects = [
        ('Project_1_Optimization', create_report_p1_en, create_report_p1_tr, create_report_p1_fa),
    ]
    
    for project, en_func, tr_func, fa_func in projects:
        report_dir = f'{project}/report'
        os.makedirs(report_dir, exist_ok=True)
        
        # English
        with open(f'{report_dir}/Report_EN.html', 'w', encoding='utf-8') as f:
            f.write(en_func())
        print(f"  ✅ {project}/report/Report_EN.html created")
        
        # Turkish
        with open(f'{report_dir}/Report_TR.html', 'w', encoding='utf-8') as f:
            f.write(tr_func())
        print(f"  ✅ {project}/report/Report_TR.html created")
        
        # Persian
        with open(f'{report_dir}/Report_FA.html', 'w', encoding='utf-8') as f:
            f.write(fa_func())
        print(f"  ✅ {project}/report/Report_FA.html created")
    
    print("\n✅ Project 1 reports created!")


if __name__ == "__main__":
    main()

