# 🔵 Proje 3: Veri Madenciliği
## Çelik Levha Hata Verilerinden Kalıp Keşfi ve Bilgi Çıkarımı

---

## 📌 Proje Bilgileri

| Öğe | Açıklama |
|-----|----------|
| **Ders** | Veri Madenciliği |
| **Konu** | Kalıp Keşfi ve Kümeleme |
| **Veri Seti** | Steel Plates Fault (OpenML ID: 1504) |
| **Seviye** | Yüksek Lisans |

---

## 🎯 Hedefler

1. Kapsamlı Keşifsel Veri Analizi (EDA) yapılması
2. Verilerdeki gizli kalıpların ve ilişkilerin keşfi
3. Boyut indirgeme tekniklerinin uygulanması (PCA, t-SNE)
4. Kümeleme yapılması ve gerçek etiketlerle karşılaştırılması
5. Verilerdeki anomalilerin tespiti

---

## 📊 Veri Seti

- **Kaynak:** [OpenML (ID: 1504)](https://www.openml.org/d/1504)
- **Örnek Sayısı:** 1.941 çelik levha hatası
- **Özellik Sayısı:** 27 sayısal özellik
- **Sınıf Sayısı:** 7 hata türü

---

## 🛠️ Teknikler

| # | Teknik | Amaç |
|---|--------|------|
| 1 | Korelasyon Analizi | Özellik ilişkileri |
| 2 | PCA | Boyut indirgeme |
| 3 | t-SNE | Görselleştirme |
| 4 | K-Means | Bölümleme kümeleme |
| 5 | Hiyerarşik Kümeleme | Dendrogram analizi |
| 6 | DBSCAN | Yoğunluk tabanlı kümeleme |
| 7 | Isolation Forest | Anomali tespiti |

---

## 📁 Proje Yapısı

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

## 🚀 Kurulum ve Kullanım

```bash
pip install -r requirements.txt
python main.py
```

---

## 📈 Beklenen Çıktılar

- Korelasyon ısı haritası
- Sınıf dağılımı grafiği
- PCA görselleştirmesi (2D, 3D)
- t-SNE görselleştirmesi
- Optimal K için Elbow grafiği
- Kümeleme dendrogramı
- Kümeleme değerlendirme metrikleri (ARI, NMI)
- Anomali tespiti sonuçları

