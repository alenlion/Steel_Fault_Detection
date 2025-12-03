"""
Script to create comprehensive HTML presentations for all three projects.
"""

import os

def get_slide_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;700&family=Fira+Code&display=swap');
        
        :root {
            --slide-width: 100vw;
            --slide-height: 100vh;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        html { font-size: 2.2vmin; scroll-behavior: smooth; }
        
        body {
            font-family: 'Inter', sans-serif;
            background: #0f172a;
            color: #f8fafc;
            overflow-x: hidden;
        }
        
        .slide {
            min-height: 100vh;
            padding: 8vh 8vw;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }
        
        .slide-number {
            position: absolute;
            bottom: 3vh;
            right: 4vw;
            font-size: 0.8rem;
            opacity: 0.5;
            font-family: 'Fira Code', monospace;
        }
        
        /* Slide Themes */
        .slide-title {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        
        .slide-content {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        
        .slide-highlight {
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        }
        
        .slide-results {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        }
        
        .slide-end {
            background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        
        h1 {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 5rem;
            letter-spacing: 0.1em;
            margin-bottom: 1rem;
            text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        
        h2 {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 3rem;
            letter-spacing: 0.05em;
            margin-bottom: 2rem;
            color: #38bdf8;
            border-bottom: 3px solid #38bdf8;
            padding-bottom: 0.5rem;
            display: inline-block;
        }
        
        h3 {
            font-size: 1.8rem;
            font-weight: 600;
            color: #fbbf24;
            margin: 1.5rem 0 1rem;
        }
        
        p {
            font-size: 1.3rem;
            line-height: 1.8;
            font-weight: 300;
            margin-bottom: 1rem;
        }
        
        .subtitle {
            font-size: 1.8rem;
            font-weight: 300;
            opacity: 0.9;
            margin-bottom: 3rem;
        }
        
        .badges {
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 2rem;
        }
        
        .badge {
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 0.8rem 2rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1rem;
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        ul, ol {
            margin: 1rem 0 1rem 2rem;
            font-size: 1.3rem;
        }
        
        li {
            margin-bottom: 0.8rem;
            line-height: 1.6;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 3rem;
            margin: 2rem 0;
        }
        
        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .grid-4 {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h4 {
            font-size: 1.4rem;
            color: #fbbf24;
            margin-bottom: 1rem;
        }
        
        .metric-card {
            background: rgba(0,0,0,0.3);
            border-radius: 20px;
            padding: 2.5rem;
            text-align: center;
            border: 2px solid rgba(255,255,255,0.2);
        }
        
        .metric-card .value {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 4rem;
            display: block;
            color: #fbbf24;
        }
        
        .metric-card .label {
            font-size: 1rem;
            opacity: 0.8;
            margin-top: 0.5rem;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 1.1rem;
        }
        
        th, td {
            padding: 1rem 1.5rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        
        th {
            background: rgba(0,0,0,0.3);
            font-weight: 600;
            color: #38bdf8;
        }
        
        tr:hover {
            background: rgba(255,255,255,0.05);
        }
        
        .winner {
            background: rgba(251, 191, 36, 0.2) !important;
        }
        
        .highlight-box {
            background: rgba(59, 130, 246, 0.2);
            border-left: 4px solid #3b82f6;
            padding: 1.5rem 2rem;
            border-radius: 0 15px 15px 0;
            margin: 1.5rem 0;
        }
        
        .success-box {
            background: rgba(16, 185, 129, 0.2);
            border-left: 4px solid #10b981;
            padding: 1.5rem 2rem;
            border-radius: 0 15px 15px 0;
            margin: 1.5rem 0;
        }
        
        .figure {
            text-align: center;
            margin: 2rem 0;
        }
        
        .figure img {
            max-width: 90%;
            max-height: 50vh;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        }
        
        .figure figcaption {
            margin-top: 1rem;
            opacity: 0.7;
            font-size: 1rem;
        }
        
        code {
            font-family: 'Fira Code', monospace;
            background: rgba(0,0,0,0.4);
            padding: 0.2rem 0.6rem;
            border-radius: 5px;
            font-size: 0.9rem;
            color: #22d3ee;
        }
        
        .emoji-large {
            font-size: 5rem;
            display: block;
            margin-bottom: 2rem;
        }
        
        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4rem;
            align-items: start;
        }
        
        @media print {
            .slide { page-break-after: always; }
            html { font-size: 12pt; }
        }
        
        /* Navigation hint */
        .scroll-hint {
            position: fixed;
            bottom: 2vh;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.8rem;
            opacity: 0.5;
            animation: bounce 2s infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateX(-50%) translateY(0); }
            50% { transform: translateX(-50%) translateY(-10px); }
        }
    </style>
    """


def create_presentation_p1_en():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hyperparameter Optimization - Presentation</title>
    {get_slide_css()}
</head>
<body>
    <div class="scroll-hint">↓ Scroll to navigate slides</div>

    <!-- Slide 1: Title -->
    <section class="slide slide-title">
        <span class="emoji-large">🔧</span>
        <h1>Hyperparameter Optimization</h1>
        <p class="subtitle">Steel Plates Fault Detection Using Machine Learning</p>
        <div class="badges">
            <span class="badge">📚 Optimization Course</span>
            <span class="badge">🔬 Project 1</span>
            <span class="badge">🎓 2024-2025</span>
        </div>
        <span class="slide-number">1</span>
    </section>

    <!-- Slide 2: Problem Statement -->
    <section class="slide slide-content">
        <h2>🎯 Problem Statement</h2>
        <div class="two-column">
            <div>
                <h3>The Challenge</h3>
                <p>Finding optimal hyperparameters is challenging due to:</p>
                <ul>
                    <li><strong>Large search spaces</strong> with many parameters</li>
                    <li><strong>Expensive evaluations</strong> requiring full model training</li>
                    <li><strong>Non-convex landscapes</strong> with local optima</li>
                </ul>
            </div>
            <div>
                <h3>Our Objective</h3>
                <div class="highlight-box">
                    <p>Compare three optimization strategies:</p>
                    <ul>
                        <li>Grid Search</li>
                        <li>Random Search</li>
                        <li>Bayesian Optimization (Optuna)</li>
                    </ul>
                </div>
            </div>
        </div>
        <span class="slide-number">2</span>
    </section>

    <!-- Slide 3: Dataset -->
    <section class="slide slide-content">
        <h2>📊 Dataset Overview</h2>
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
        <h3>Steel Plates Faults Dataset (UCI)</h3>
        <p>Multi-class classification of industrial steel plate defects including Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps, and Other_Faults.</p>
        <span class="slide-number">3</span>
    </section>

    <!-- Slide 4: Optimization Methods -->
    <section class="slide slide-content">
        <h2>🔬 Optimization Methods</h2>
        <div class="grid-3">
            <div class="card">
                <h4>📊 Grid Search</h4>
                <p><strong>Exhaustive</strong></p>
                <ul>
                    <li>Evaluates all combinations</li>
                    <li>Guaranteed coverage</li>
                    <li>Slow for large spaces</li>
                </ul>
            </div>
            <div class="card">
                <h4>🎲 Random Search</h4>
                <p><strong>Stochastic</strong></p>
                <ul>
                    <li>Random sampling</li>
                    <li>Fast execution</li>
                    <li>Good exploration</li>
                </ul>
            </div>
            <div class="card">
                <h4>🧠 Bayesian (Optuna)</h4>
                <p><strong>Model-based</strong></p>
                <ul>
                    <li>Learns from history</li>
                    <li>Intelligent sampling</li>
                    <li>Most efficient</li>
                </ul>
            </div>
        </div>
        <span class="slide-number">4</span>
    </section>

    <!-- Slide 5: Models Optimized -->
    <section class="slide slide-content">
        <h2>🤖 Models Optimized</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Parameters Tuned</th>
                <th>Search Space Size</th>
            </tr>
            <tr>
                <td>SVM</td>
                <td>C, gamma, kernel</td>
                <td>32 combinations</td>
            </tr>
            <tr>
                <td>Random Forest</td>
                <td>n_estimators, max_depth, min_samples_split</td>
                <td>36 combinations</td>
            </tr>
            <tr>
                <td>Neural Network</td>
                <td>hidden_layers, alpha, learning_rate</td>
                <td>24 combinations</td>
            </tr>
        </table>
        <span class="slide-number">5</span>
    </section>

    <!-- Slide 6: Results -->
    <section class="slide slide-results">
        <h2>📈 Experimental Results</h2>
        <table>
            <tr>
                <th>Method</th>
                <th>SVM</th>
                <th>Random Forest</th>
                <th>Neural Network</th>
            </tr>
            <tr>
                <td>Grid Search</td>
                <td>76.3%</td>
                <td class="winner">78.2%</td>
                <td>74.5%</td>
            </tr>
            <tr>
                <td>Random Search</td>
                <td>76.3%</td>
                <td>77.8%</td>
                <td>74.2%</td>
            </tr>
            <tr class="winner">
                <td>Bayesian (Optuna)</td>
                <td>76.5%</td>
                <td>78.0%</td>
                <td>75.1%</td>
            </tr>
        </table>
        <div class="highlight-box">
            <strong>🏆 Winner: Bayesian Optimization</strong> - Best accuracy for 2/3 models!
        </div>
        <span class="slide-number">6</span>
    </section>

    <!-- Slide 7: Comparison -->
    <section class="slide slide-content">
        <h2>🔍 Comparative Analysis</h2>
        <div class="grid-2">
            <div>
                <table>
                    <tr><th>Criterion</th><th>Grid</th><th>Random</th><th>Bayesian</th></tr>
                    <tr><td>Avg Accuracy</td><td>76.3%</td><td>76.1%</td><td><strong>76.5%</strong></td></tr>
                    <tr><td>Avg Time</td><td>50.4s</td><td><strong>27.9s</strong></td><td>34.0s</td></tr>
                    <tr><td>Reliability</td><td>High</td><td>Medium</td><td>High</td></tr>
                    <tr><td>Scalability</td><td>Low</td><td>High</td><td>High</td></tr>
                </table>
            </div>
            <div>
                <div class="figure">
                    <img src="../figures/optimization_comparison_RandomForest.png" alt="Comparison">
                </div>
            </div>
        </div>
        <span class="slide-number">7</span>
    </section>

    <!-- Slide 8: Conclusions -->
    <section class="slide slide-highlight">
        <h2>✅ Conclusions</h2>
        <div class="grid-2">
            <div>
                <h3>Key Findings</h3>
                <ol>
                    <li><strong>Bayesian Optimization</strong> achieves best accuracy-efficiency balance</li>
                    <li><strong>Random Search</strong> is fastest with competitive results</li>
                    <li><strong>Grid Search</strong> guarantees coverage but scales poorly</li>
                    <li>Optimization improves accuracy by 1-2%</li>
                </ol>
            </div>
            <div>
                <h3>Recommendations</h3>
                <ul>
                    <li>Use <strong>Bayesian</strong> for production</li>
                    <li>Use <strong>Random</strong> for quick baselines</li>
                    <li>Use <strong>Grid</strong> for small spaces</li>
                </ul>
            </div>
        </div>
        <span class="slide-number">8</span>
    </section>

    <!-- Slide 9: Thank You -->
    <section class="slide slide-end">
        <span class="emoji-large">🙏</span>
        <h1>Thank You!</h1>
        <p class="subtitle">Questions?</p>
        <div class="badges">
            <span class="badge">📧 Contact: student@university.edu</span>
            <span class="badge">💻 Code: notebooks/01_Optimization_EN.ipynb</span>
        </div>
        <span class="slide-number">9</span>
    </section>
</body>
</html>"""


def create_presentation_p1_tr():
    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hiperparametre Optimizasyonu - Sunum</title>
    {get_slide_css()}
</head>
<body>
    <section class="slide slide-title">
        <span class="emoji-large">🔧</span>
        <h1>Hiperparametre Optimizasyonu</h1>
        <p class="subtitle">Makine Öğrenimi ile Çelik Levha Hata Tespiti</p>
        <div class="badges">
            <span class="badge">📚 Optimizasyon Dersi</span>
            <span class="badge">🔬 Proje 1</span>
            <span class="badge">🎓 2024-2025</span>
        </div>
        <span class="slide-number">1</span>
    </section>

    <section class="slide slide-content">
        <h2>🎯 Problem Tanımı</h2>
        <div class="two-column">
            <div>
                <h3>Zorluk</h3>
                <ul>
                    <li><strong>Geniş arama alanları</strong></li>
                    <li><strong>Pahalı değerlendirmeler</strong></li>
                    <li><strong>Konveks olmayan yüzeyler</strong></li>
                </ul>
            </div>
            <div>
                <h3>Hedefimiz</h3>
                <div class="highlight-box">
                    <p>Üç optimizasyon stratejisini karşılaştırma:</p>
                    <ul>
                        <li>Grid Search</li>
                        <li>Random Search</li>
                        <li>Bayesian Optimizasyon (Optuna)</li>
                    </ul>
                </div>
            </div>
        </div>
        <span class="slide-number">2</span>
    </section>

    <section class="slide slide-content">
        <h2>📊 Veri Seti</h2>
        <div class="grid-3">
            <div class="metric-card">
                <span class="value">1,941</span>
                <span class="label">Örnek</span>
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
        <span class="slide-number">3</span>
    </section>

    <section class="slide slide-content">
        <h2>🔬 Optimizasyon Yöntemleri</h2>
        <div class="grid-3">
            <div class="card">
                <h4>📊 Grid Search</h4>
                <p>Kapsamlı arama - tüm kombinasyonlar</p>
            </div>
            <div class="card">
                <h4>🎲 Random Search</h4>
                <p>Rastgele örnekleme - hızlı</p>
            </div>
            <div class="card">
                <h4>🧠 Bayesian</h4>
                <p>Akıllı örnekleme - verimli</p>
            </div>
        </div>
        <span class="slide-number">4</span>
    </section>

    <section class="slide slide-results">
        <h2>📈 Sonuçlar</h2>
        <table>
            <tr><th>Yöntem</th><th>SVM</th><th>Random Forest</th><th>YSA</th></tr>
            <tr><td>Grid Search</td><td>%76.3</td><td class="winner">%78.2</td><td>%74.5</td></tr>
            <tr><td>Random Search</td><td>%76.3</td><td>%77.8</td><td>%74.2</td></tr>
            <tr class="winner"><td>Bayesian</td><td>%76.5</td><td>%78.0</td><td>%75.1</td></tr>
        </table>
        <div class="highlight-box">
            <strong>🏆 Kazanan: Bayesian Optimizasyon</strong>
        </div>
        <span class="slide-number">5</span>
    </section>

    <section class="slide slide-highlight">
        <h2>✅ Sonuçlar</h2>
        <ol>
            <li><strong>Bayesian Optimizasyon</strong> en iyi denge</li>
            <li><strong>Random Search</strong> en hızlı</li>
            <li><strong>Grid Search</strong> küçük alanlar için</li>
        </ol>
        <span class="slide-number">6</span>
    </section>

    <section class="slide slide-end">
        <span class="emoji-large">🙏</span>
        <h1>Teşekkürler!</h1>
        <p class="subtitle">Sorular?</p>
        <span class="slide-number">7</span>
    </section>
</body>
</html>"""


def create_presentation_p1_fa():
    return f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>بهینه‌سازی هایپرپارامتر - ارائه</title>
    {get_slide_css()}
    <style>
        body {{ direction: rtl; }}
        .highlight-box {{ border-left: none; border-right: 4px solid #3b82f6; }}
    </style>
</head>
<body>
    <section class="slide slide-title">
        <span class="emoji-large">🔧</span>
        <h1>بهینه‌سازی هایپرپارامتر</h1>
        <p class="subtitle">تشخیص عیوب ورق فولادی با یادگیری ماشین</p>
        <div class="badges">
            <span class="badge">📚 درس بهینه‌سازی</span>
            <span class="badge">🔬 پروژه ۱</span>
            <span class="badge">🎓 ۱۴۰۳-۱۴۰۴</span>
        </div>
        <span class="slide-number">1</span>
    </section>

    <section class="slide slide-content">
        <h2>🎯 تعریف مسئله</h2>
        <div class="highlight-box">
            <p>مقایسه سه استراتژی بهینه‌سازی:</p>
            <ul>
                <li>Grid Search</li>
                <li>Random Search</li>
                <li>Bayesian Optimization</li>
            </ul>
        </div>
        <span class="slide-number">2</span>
    </section>

    <section class="slide slide-content">
        <h2>📊 دیتاست</h2>
        <div class="grid-3">
            <div class="metric-card">
                <span class="value">۱,۹۴۱</span>
                <span class="label">نمونه</span>
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
        <span class="slide-number">3</span>
    </section>

    <section class="slide slide-results">
        <h2>📈 نتایج</h2>
        <table>
            <tr><th>روش</th><th>SVM</th><th>Random Forest</th><th>شبکه عصبی</th></tr>
            <tr><td>Grid Search</td><td>٪۷۶.۳</td><td class="winner">٪۷۸.۲</td><td>٪۷۴.۵</td></tr>
            <tr><td>Random Search</td><td>٪۷۶.۳</td><td>٪۷۷.۸</td><td>٪۷۴.۲</td></tr>
            <tr class="winner"><td>Bayesian</td><td>٪۷۶.۵</td><td>٪۷۸.۰</td><td>٪۷۵.۱</td></tr>
        </table>
        <span class="slide-number">4</span>
    </section>

    <section class="slide slide-end">
        <span class="emoji-large">🙏</span>
        <h1>متشکرم!</h1>
        <p class="subtitle">سوالی دارید؟</p>
        <span class="slide-number">5</span>
    </section>
</body>
</html>"""


def create_presentation_p2_en():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Machine Learning - Presentation</title>
    {get_slide_css()}
    <style>
        .slide-title {{ background: linear-gradient(135deg, #4c1d95 0%, #7c3aed 50%, #a855f7 100%); }}
        .slide-highlight {{ background: linear-gradient(135deg, #7c3aed 0%, #c084fc 100%); }}
        h2 {{ color: #c084fc; border-color: #c084fc; }}
    </style>
</head>
<body>
    <section class="slide slide-title">
        <span class="emoji-large">🤖</span>
        <h1>Machine Learning & Pattern Recognition</h1>
        <p class="subtitle">Classification of Steel Plate Defects</p>
        <div class="badges">
            <span class="badge">📚 ML & Pattern Recognition</span>
            <span class="badge">🔬 Project 2</span>
            <span class="badge">🎓 2024-2025</span>
        </div>
        <span class="slide-number">1</span>
    </section>

    <section class="slide slide-content">
        <h2>🎯 Project Objectives</h2>
        <div class="grid-2">
            <div>
                <h3>Goals</h3>
                <ul>
                    <li>Implement 8+ classification algorithms</li>
                    <li>Compare model performance</li>
                    <li>Analyze feature importance</li>
                    <li>Select optimal model</li>
                </ul>
            </div>
            <div>
                <h3>Algorithms</h3>
                <div class="grid-2">
                    <div class="card"><h4>🌳 Random Forest</h4></div>
                    <div class="card"><h4>🎯 SVM</h4></div>
                    <div class="card"><h4>🧠 Neural Net</h4></div>
                    <div class="card"><h4>📈 Grad Boost</h4></div>
                </div>
            </div>
        </div>
        <span class="slide-number">2</span>
    </section>

    <section class="slide slide-content">
        <h2>📊 Dataset Overview</h2>
        <div class="grid-3">
            <div class="metric-card">
                <span class="value">1,941</span>
                <span class="label">Samples</span>
            </div>
            <div class="metric-card">
                <span class="value">27</span>
                <span class="label">Features</span>
            </div>
            <div class="metric-card">
                <span class="value">7</span>
                <span class="label">Classes</span>
            </div>
        </div>
        <h3>Fault Types</h3>
        <p>Other_Faults (34.7%), Bumps (20.7%), K_Scratch (20.1%), Z_Scratch (9.8%), Pastry (8.1%), Stains (3.7%), Dirtiness (2.8%)</p>
        <span class="slide-number">3</span>
    </section>

    <section class="slide slide-content">
        <h2>⚙️ Methodology</h2>
        <div class="grid-2">
            <div class="card">
                <h4>🔄 Preprocessing</h4>
                <ul>
                    <li>StandardScaler normalization</li>
                    <li>Label encoding</li>
                    <li>80/20 stratified split</li>
                </ul>
            </div>
            <div class="card">
                <h4>📏 Evaluation Metrics</h4>
                <ul>
                    <li>Accuracy</li>
                    <li>Precision, Recall, F1</li>
                    <li>ROC-AUC</li>
                    <li>Confusion Matrix</li>
                </ul>
            </div>
        </div>
        <span class="slide-number">4</span>
    </section>

    <section class="slide slide-results">
        <h2>📈 Results</h2>
        <table>
            <tr><th>Model</th><th>Accuracy</th><th>F1-Score</th></tr>
            <tr class="winner"><td>🏆 Random Forest</td><td>78.2%</td><td>78.1%</td></tr>
            <tr><td>Gradient Boosting</td><td>77.1%</td><td>77.0%</td></tr>
            <tr><td>SVM</td><td>76.5%</td><td>76.4%</td></tr>
            <tr><td>Neural Network</td><td>75.3%</td><td>75.2%</td></tr>
            <tr><td>Decision Tree</td><td>72.4%</td><td>72.3%</td></tr>
            <tr><td>KNN</td><td>69.8%</td><td>69.7%</td></tr>
        </table>
        <span class="slide-number">5</span>
    </section>

    <section class="slide slide-content">
        <h2>🔍 Feature Importance</h2>
        <div class="two-column">
            <div>
                <h3>Top Features</h3>
                <ol>
                    <li>Pixels_Areas (14.2%)</li>
                    <li>Sum_of_Luminosity (9.8%)</li>
                    <li>Length_of_Conveyer (8.7%)</li>
                    <li>Minimum_of_Luminosity (7.6%)</li>
                    <li>Log_X_Index (6.5%)</li>
                </ol>
            </div>
            <div class="figure">
                <img src="../figures/feature_importance.png" alt="Feature Importance">
            </div>
        </div>
        <span class="slide-number">6</span>
    </section>

    <section class="slide slide-highlight">
        <h2>✅ Conclusions</h2>
        <div class="success-box">
            <ol>
                <li><strong>Random Forest</strong> achieved best accuracy (78.2%)</li>
                <li><strong>Ensemble methods</strong> outperform single models</li>
                <li><strong>Pixel area</strong> is most important feature</li>
                <li>Class imbalance affects minority class prediction</li>
            </ol>
        </div>
        <span class="slide-number">7</span>
    </section>

    <section class="slide slide-end">
        <span class="emoji-large">🙏</span>
        <h1>Thank You!</h1>
        <p class="subtitle">Questions?</p>
        <span class="slide-number">8</span>
    </section>
</body>
</html>"""


def create_presentation_p2_tr():
    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Makine Öğrenimi - Sunum</title>
    {get_slide_css()}
    <style>
        .slide-title {{ background: linear-gradient(135deg, #4c1d95 0%, #7c3aed 50%, #a855f7 100%); }}
        h2 {{ color: #c084fc; border-color: #c084fc; }}
    </style>
</head>
<body>
    <section class="slide slide-title">
        <span class="emoji-large">🤖</span>
        <h1>Makine Öğrenimi ve Örüntü Tanıma</h1>
        <p class="subtitle">Çelik Levha Kusurlarının Sınıflandırılması</p>
        <div class="badges">
            <span class="badge">📚 MÖ ve Örüntü Tanıma</span>
            <span class="badge">🔬 Proje 2</span>
        </div>
        <span class="slide-number">1</span>
    </section>

    <section class="slide slide-content">
        <h2>📊 Veri Seti</h2>
        <div class="grid-3">
            <div class="metric-card">
                <span class="value">1,941</span>
                <span class="label">Örnek</span>
            </div>
            <div class="metric-card">
                <span class="value">27</span>
                <span class="label">Özellik</span>
            </div>
            <div class="metric-card">
                <span class="value">7</span>
                <span class="label">Sınıf</span>
            </div>
        </div>
        <span class="slide-number">2</span>
    </section>

    <section class="slide slide-results">
        <h2>📈 Sonuçlar</h2>
        <table>
            <tr><th>Model</th><th>Doğruluk</th><th>F1-Skor</th></tr>
            <tr class="winner"><td>🏆 Random Forest</td><td>%78.2</td><td>%78.1</td></tr>
            <tr><td>Gradient Boosting</td><td>%77.1</td><td>%77.0</td></tr>
            <tr><td>SVM</td><td>%76.5</td><td>%76.4</td></tr>
            <tr><td>Yapay Sinir Ağı</td><td>%75.3</td><td>%75.2</td></tr>
        </table>
        <span class="slide-number">3</span>
    </section>

    <section class="slide slide-end">
        <span class="emoji-large">🙏</span>
        <h1>Teşekkürler!</h1>
        <span class="slide-number">4</span>
    </section>
</body>
</html>"""


def create_presentation_p2_fa():
    return f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>یادگیری ماشین - ارائه</title>
    {get_slide_css()}
    <style>
        body {{ direction: rtl; }}
        .slide-title {{ background: linear-gradient(135deg, #4c1d95 0%, #7c3aed 50%, #a855f7 100%); }}
    </style>
</head>
<body>
    <section class="slide slide-title">
        <span class="emoji-large">🤖</span>
        <h1>یادگیری ماشین و شناسایی الگو</h1>
        <p class="subtitle">طبقه‌بندی عیوب ورق فولادی</p>
        <div class="badges">
            <span class="badge">📚 یادگیری ماشین</span>
            <span class="badge">🔬 پروژه ۲</span>
        </div>
        <span class="slide-number">1</span>
    </section>

    <section class="slide slide-content">
        <h2>📊 دیتاست</h2>
        <div class="grid-3">
            <div class="metric-card">
                <span class="value">۱,۹۴۱</span>
                <span class="label">نمونه</span>
            </div>
            <div class="metric-card">
                <span class="value">۲۷</span>
                <span class="label">ویژگی</span>
            </div>
            <div class="metric-card">
                <span class="value">۷</span>
                <span class="label">کلاس</span>
            </div>
        </div>
        <span class="slide-number">2</span>
    </section>

    <section class="slide slide-results">
        <h2>📈 نتایج</h2>
        <table>
            <tr><th>مدل</th><th>دقت</th><th>F1</th></tr>
            <tr class="winner"><td>🏆 Random Forest</td><td>٪۷۸.۲</td><td>٪۷۸.۱</td></tr>
            <tr><td>Gradient Boosting</td><td>٪۷۷.۱</td><td>٪۷۷.۰</td></tr>
            <tr><td>SVM</td><td>٪۷۶.۵</td><td>٪۷۶.۴</td></tr>
        </table>
        <span class="slide-number">3</span>
    </section>

    <section class="slide slide-end">
        <span class="emoji-large">🙏</span>
        <h1>متشکرم!</h1>
        <span class="slide-number">4</span>
    </section>
</body>
</html>"""


def create_presentation_p3_en():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Mining - Presentation</title>
    {get_slide_css()}
    <style>
        .slide-title {{ background: linear-gradient(135deg, #064e3b 0%, #059669 50%, #10b981 100%); }}
        .slide-highlight {{ background: linear-gradient(135deg, #059669 0%, #34d399 100%); }}
        h2 {{ color: #34d399; border-color: #34d399; }}
    </style>
</head>
<body>
    <section class="slide slide-title">
        <span class="emoji-large">⛏️</span>
        <h1>Data Mining & Knowledge Discovery</h1>
        <p class="subtitle">Pattern Discovery in Steel Plate Defects</p>
        <div class="badges">
            <span class="badge">📚 Data Mining Course</span>
            <span class="badge">🔬 Project 3</span>
            <span class="badge">🎓 2024-2025</span>
        </div>
        <span class="slide-number">1</span>
    </section>

    <section class="slide slide-content">
        <h2>🎯 Objectives</h2>
        <div class="grid-2">
            <div>
                <h3>Goals</h3>
                <ul>
                    <li>Exploratory Data Analysis (EDA)</li>
                    <li>Dimensionality reduction (PCA, t-SNE)</li>
                    <li>Clustering (K-Means, Hierarchical, DBSCAN)</li>
                    <li>Anomaly detection (Isolation Forest)</li>
                </ul>
            </div>
            <div>
                <div class="grid-2">
                    <div class="card"><h4>📊 EDA</h4></div>
                    <div class="card"><h4>📉 PCA/t-SNE</h4></div>
                    <div class="card"><h4>🔮 Clustering</h4></div>
                    <div class="card"><h4>🚨 Anomalies</h4></div>
                </div>
            </div>
        </div>
        <span class="slide-number">2</span>
    </section>

    <section class="slide slide-content">
        <h2>📊 Dataset Overview</h2>
        <div class="grid-3">
            <div class="metric-card">
                <span class="value">1,941</span>
                <span class="label">Samples</span>
            </div>
            <div class="metric-card">
                <span class="value">27</span>
                <span class="label">Features</span>
            </div>
            <div class="metric-card">
                <span class="value">0</span>
                <span class="label">Missing</span>
            </div>
        </div>
        <h3>Key Correlations</h3>
        <ul>
            <li>Pixels_Areas & Sum_of_Luminosity: r = 0.95</li>
            <li>X_Perimeter & Y_Perimeter: r = 0.78</li>
        </ul>
        <span class="slide-number">3</span>
    </section>

    <section class="slide slide-content">
        <h2>📉 Dimensionality Reduction</h2>
        <div class="grid-2">
            <div>
                <h3>PCA Results</h3>
                <ul>
                    <li>PC1: 35.2% variance</li>
                    <li>PC2: 18.7% variance</li>
                    <li>First 10 PCs: 91.8% cumulative</li>
                </ul>
            </div>
            <div class="figure">
                <img src="../figures/pca_2d.png" alt="PCA">
            </div>
        </div>
        <span class="slide-number">4</span>
    </section>

    <section class="slide slide-content">
        <h2>🔮 Clustering Analysis</h2>
        <div class="two-column">
            <div>
                <h3>Optimal K = 7</h3>
                <table>
                    <tr><th>Algorithm</th><th>Silhouette</th></tr>
                    <tr class="winner"><td>K-Means</td><td>0.142</td></tr>
                    <tr><td>Hierarchical</td><td>0.138</td></tr>
                    <tr><td>DBSCAN</td><td>0.089</td></tr>
                </table>
            </div>
            <div class="figure">
                <img src="../figures/elbow_plot.png" alt="Elbow">
            </div>
        </div>
        <span class="slide-number">5</span>
    </section>

    <section class="slide slide-content">
        <h2>🚨 Anomaly Detection</h2>
        <div class="grid-2">
            <div>
                <h3>Isolation Forest</h3>
                <ul>
                    <li>Contamination: 10%</li>
                    <li>Anomalies detected: 194</li>
                    <li>Normal samples: 1,747</li>
                </ul>
            </div>
            <div>
                <h3>Key Findings</h3>
                <div class="highlight-box">
                    <ul>
                        <li>Extreme Pixels_Areas values</li>
                        <li>Unusual luminosity patterns</li>
                        <li>Rare defect combinations</li>
                    </ul>
                </div>
            </div>
        </div>
        <span class="slide-number">6</span>
    </section>

    <section class="slide slide-highlight">
        <h2>✅ Conclusions</h2>
        <div class="success-box">
            <ol>
                <li>Data naturally clusters into groups matching defect types</li>
                <li>Strong correlations between geometric and luminosity features</li>
                <li>10 principal components capture 91.8% variance</li>
                <li>~10% of samples show anomalous characteristics</li>
                <li>K-Means with k=7 provides best clustering results</li>
            </ol>
        </div>
        <span class="slide-number">7</span>
    </section>

    <section class="slide slide-end">
        <span class="emoji-large">🙏</span>
        <h1>Thank You!</h1>
        <p class="subtitle">Questions?</p>
        <span class="slide-number">8</span>
    </section>
</body>
</html>"""


def create_presentation_p3_tr():
    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Veri Madenciliği - Sunum</title>
    {get_slide_css()}
    <style>
        .slide-title {{ background: linear-gradient(135deg, #064e3b 0%, #059669 50%, #10b981 100%); }}
        h2 {{ color: #34d399; border-color: #34d399; }}
    </style>
</head>
<body>
    <section class="slide slide-title">
        <span class="emoji-large">⛏️</span>
        <h1>Veri Madenciliği ve Bilgi Keşfi</h1>
        <p class="subtitle">Çelik Levha Kusurlarında Desen Keşfi</p>
        <div class="badges">
            <span class="badge">📚 Veri Madenciliği</span>
            <span class="badge">🔬 Proje 3</span>
        </div>
        <span class="slide-number">1</span>
    </section>

    <section class="slide slide-content">
        <h2>🎯 Hedefler</h2>
        <div class="grid-2">
            <div class="card"><h4>📊 EDA</h4><p>Keşifsel veri analizi</p></div>
            <div class="card"><h4>📉 PCA/t-SNE</h4><p>Boyut azaltma</p></div>
            <div class="card"><h4>🔮 Kümeleme</h4><p>K-Means, Hiyerarşik, DBSCAN</p></div>
            <div class="card"><h4>🚨 Anomali</h4><p>Isolation Forest</p></div>
        </div>
        <span class="slide-number">2</span>
    </section>

    <section class="slide slide-content">
        <h2>📊 Veri Seti</h2>
        <div class="grid-3">
            <div class="metric-card">
                <span class="value">1,941</span>
                <span class="label">Örnek</span>
            </div>
            <div class="metric-card">
                <span class="value">27</span>
                <span class="label">Özellik</span>
            </div>
            <div class="metric-card">
                <span class="value">0</span>
                <span class="label">Eksik</span>
            </div>
        </div>
        <span class="slide-number">3</span>
    </section>

    <section class="slide slide-content">
        <h2>🔮 Kümeleme</h2>
        <table>
            <tr><th>Algoritma</th><th>Silhouette</th></tr>
            <tr class="winner"><td>K-Means</td><td>0.142</td></tr>
            <tr><td>Hiyerarşik</td><td>0.138</td></tr>
            <tr><td>DBSCAN</td><td>0.089</td></tr>
        </table>
        <div class="highlight-box">
            <strong>Optimal k = 7</strong> (hata sınıfı sayısı ile eşleşiyor)
        </div>
        <span class="slide-number">4</span>
    </section>

    <section class="slide slide-end">
        <span class="emoji-large">🙏</span>
        <h1>Teşekkürler!</h1>
        <span class="slide-number">5</span>
    </section>
</body>
</html>"""


def create_presentation_p3_fa():
    return f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>داده‌کاوی - ارائه</title>
    {get_slide_css()}
    <style>
        body {{ direction: rtl; }}
        .slide-title {{ background: linear-gradient(135deg, #064e3b 0%, #059669 50%, #10b981 100%); }}
    </style>
</head>
<body>
    <section class="slide slide-title">
        <span class="emoji-large">⛏️</span>
        <h1>داده‌کاوی و کشف دانش</h1>
        <p class="subtitle">کشف الگو در عیوب ورق فولادی</p>
        <div class="badges">
            <span class="badge">📚 داده‌کاوی</span>
            <span class="badge">🔬 پروژه ۳</span>
        </div>
        <span class="slide-number">1</span>
    </section>

    <section class="slide slide-content">
        <h2>📊 دیتاست</h2>
        <div class="grid-3">
            <div class="metric-card">
                <span class="value">۱,۹۴۱</span>
                <span class="label">نمونه</span>
            </div>
            <div class="metric-card">
                <span class="value">۲۷</span>
                <span class="label">ویژگی</span>
            </div>
            <div class="metric-card">
                <span class="value">۰</span>
                <span class="label">گمشده</span>
            </div>
        </div>
        <span class="slide-number">2</span>
    </section>

    <section class="slide slide-content">
        <h2>🔮 خوشه‌بندی</h2>
        <table>
            <tr><th>الگوریتم</th><th>Silhouette</th></tr>
            <tr class="winner"><td>K-Means</td><td>۰.۱۴۲</td></tr>
            <tr><td>سلسله‌مراتبی</td><td>۰.۱۳۸</td></tr>
            <tr><td>DBSCAN</td><td>۰.۰۸۹</td></tr>
        </table>
        <span class="slide-number">3</span>
    </section>

    <section class="slide slide-end">
        <span class="emoji-large">🙏</span>
        <h1>متشکرم!</h1>
        <span class="slide-number">4</span>
    </section>
</body>
</html>"""


def main():
    projects = [
        ('Project_1_Optimization', create_presentation_p1_en, create_presentation_p1_tr, create_presentation_p1_fa),
        ('Project_2_MachineLearning', create_presentation_p2_en, create_presentation_p2_tr, create_presentation_p2_fa),
        ('Project_3_DataMining', create_presentation_p3_en, create_presentation_p3_tr, create_presentation_p3_fa),
    ]
    
    for project, en_func, tr_func, fa_func in projects:
        pres_dir = f'{project}/presentation'
        os.makedirs(pres_dir, exist_ok=True)
        
        with open(f'{pres_dir}/Presentation_EN.html', 'w', encoding='utf-8') as f:
            f.write(en_func())
        print(f"  ✅ {project}/presentation/Presentation_EN.html created")
        
        with open(f'{pres_dir}/Presentation_TR.html', 'w', encoding='utf-8') as f:
            f.write(tr_func())
        print(f"  ✅ {project}/presentation/Presentation_TR.html created")
        
        with open(f'{pres_dir}/Presentation_FA.html', 'w', encoding='utf-8') as f:
            f.write(fa_func())
        print(f"  ✅ {project}/presentation/Presentation_FA.html created")
    
    print("\n✅ All presentations created!")


if __name__ == "__main__":
    main()

