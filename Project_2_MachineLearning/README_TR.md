# 🟢 Proje 2: Makine Öğrenimi ve Örüntü Tanıma
## Makine Öğrenimi Algoritmaları ile Çelik Levha Hatalarının Çok Sınıflı Sınıflandırılması

---

## 📌 Proje Bilgileri

| Öğe | Açıklama |
|-----|----------|
| **Ders** | Makine Öğrenimi ve Örüntü Tanıma |
| **Konu** | Çok Sınıflı Sınıflandırma |
| **Veri Seti** | Steel Plates Fault (OpenML ID: 1504) |
| **Seviye** | Yüksek Lisans |

---

## 🎯 Hedefler

1. Farklı sınıflandırma algoritmalarının karşılaştırılması
2. Önemli özellikleri belirlemek için özellik seçimi
3. Çeşitli metriklerle model performansının değerlendirilmesi
4. Karışıklık matrisi ve ROC eğrilerinin analizi

---

## 📊 Veri Seti

- **Kaynak:** [OpenML (ID: 1504)](https://www.openml.org/d/1504)
- **Örnek Sayısı:** 1.941 çelik levha hatası
- **Özellik Sayısı:** 27 sayısal özellik
- **Sınıf Sayısı:** 7 hata türü

---

## 🛠️ Modeller

| # | Model | Tür |
|---|-------|-----|
| 1 | Lojistik Regresyon | Doğrusal |
| 2 | K-En Yakın Komşu | Mesafe tabanlı |
| 3 | Karar Ağacı | Ağaç tabanlı |
| 4 | Rastgele Orman | Topluluk |
| 5 | Gradient Boosting | Topluluk |
| 6 | XGBoost | Topluluk |
| 7 | SVM (RBF) | Çekirdek tabanlı |
| 8 | Sinir Ağı (MLP) | Derin Öğrenme |

---

## 📁 Proje Yapısı

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

## 🚀 Kurulum ve Kullanım

```bash
pip install -r requirements.txt
python main.py
```

---

## 📈 Beklenen Çıktılar

- Model karşılaştırma tablosu (Accuracy, Precision, Recall, F1, ROC-AUC)
- En iyi model için karışıklık matrisi
- Tüm modeller için ROC eğrileri
- Özellik önemi grafiği
- Sınıflandırma raporu

