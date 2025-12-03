# 🔴 Proje 1: Optimizasyon
## Çelik Hata Sınıflandırmasında Hiperparametre Ayarı için Optimizasyon Algoritmalarının Karşılaştırılması

---

## 📌 Proje Bilgileri

| Öğe | Açıklama |
|-----|----------|
| **Ders** | Optimizasyon |
| **Konu** | Hiperparametre Optimizasyonu |
| **Veri Seti** | Steel Plates Fault (OpenML ID: 1504) |
| **Seviye** | Yüksek Lisans |

---

## 🎯 Hedefler

1. Farklı hiperparametre optimizasyon yöntemlerinin karşılaştırılması
2. Algoritmaların yakınsama ve verimliliğinin analizi
3. Makine öğrenimi modelleri için en iyi ayarların bulunması
4. Her yöntemin zaman karmaşıklığının değerlendirilmesi

---

## 📊 Veri Seti

- **Kaynak:** [OpenML (ID: 1504)](https://www.openml.org/d/1504)
- **Örnek Sayısı:** 1.941 çelik levha hatası
- **Özellik Sayısı:** 27 sayısal özellik
- **Sınıf Sayısı:** 7 hata türü

---

## 🛠️ Yöntemler

| # | Yöntem | Kütüphane |
|---|--------|-----------|
| 1 | Izgara Araması | `sklearn.GridSearchCV` |
| 2 | Rastgele Arama | `sklearn.RandomizedSearchCV` |
| 3 | Bayes Optimizasyonu | `Optuna` |

---

## 📁 Proje Yapısı

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

## 🚀 Kurulum ve Kullanım

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Projeyi çalıştır
python main.py
```

---

## 📈 Beklenen Çıktılar

- Optimizasyon algoritmaları karşılaştırma tablosu
- Yakınsama grafikleri
- Çalışma süresi karşılaştırma grafiği
- SVM, Random Forest, Sinir Ağı için en iyi hiperparametreler
- Duyarlılık analizi

---

## 📚 Kaynaklar

1. Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization.
2. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization.
3. Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework.

