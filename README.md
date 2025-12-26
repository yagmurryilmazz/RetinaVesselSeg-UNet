<div align="center">

# 👁️ DeepRetina  
### Retinal Damar Segmentasyonu / Retinal Vessel Segmentation

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-informational)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-informational)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2B-informational)](#)
[![Task](https://img.shields.io/badge/Task-Segmentation-blue)](#)
[![Dataset](https://img.shields.io/badge/Dataset-DRIVE-green)](#)

Yüksek hassasiyetli **retinal damar ağı segmentasyonu** için **U-Net tabanlı** derin öğrenme projesi.  
**Sistemik hastalıkların (diyabetik retinopati, hipertansiyon vb.) dijital biyobelirteçlerini** analiz etmek için güvenilir bir ön-aşama segmentasyon sağlar.

</div>

---

## 🔎 İçindekiler
- [🇹🇷 Proje Özeti (TR)](#-proje-özeti-tr)
- [🇺🇸 Project Overview (EN)](#-project-overview-en)
- [🧠 Metodoloji](#-metodoloji)
- [🖼️ Ön İşleme](#️-ön-işleme)
- [📈 Performans](#-performans)
- [🖼️ Örnek Çıktılar](#️-örnek-çıktılar)
- [📁 Klasör Yapısı](#-klasör-yapısı)
- [🧰 Kurulum](#-kurulum)
- [🚀 Çalıştırma](#-çalıştırma)
- [📌 Notlar](#-notlar)
- [📝 Atıf / Citation](#-atıf--citation)

---

## 🧰 Kurulum

### 1) Veri Seti (DRIVE)
DRIVE dataset (Kaggle) üzerinden indirilebilir:

- **DRIVE Dataset Download (Kaggle):** `KAGGLE_LINKINI_BURAYA_YAPISTIR`

> Örnek format:  
> `https://www.kaggle.com/datasets/...`

### 2) Bağımlılıklar
```bash
pip install tensorflow opencv-python numpy pillow scikit-learn

## 🚀 Çalıştırma
1. `main.ipynb` dosyasını aç  
2. Dosya yollarını kendi ortamına göre güncelle  
3. Hücreleri sırasıyla çalıştır  
4. Çıktılar:
   - Model: `Modeller/`
   - Tahminler: `Tahminler/`

---

## 🇹🇷 Proje Özeti (TR)
Bu projede, **fundus anjiyografi** görüntülerinden damar ağını otomatik ayırt eden, yüksek hassasiyetli bir **derin öğrenme segmentasyon modeli** geliştirdim.  
Amaç; klinik karar destek sistemlerinde kullanılabilecek, **güvenilir bir damar segmentasyonu** üretmektir.

---

## 🇺🇸 Project Overview (EN)
In this project, I developed a high-precision **deep learning model** to automatically segment the vascular network from **fundus angiography** images.  
The goal is to provide a reliable pre-processing segmentation for **clinical decision support** and digital biomarker analysis.

---

## 🧠 Metodoloji
### 🛰️ Mimari: U-Net
Medikal görüntü segmentasyonunda altın standartlardan biri olan **U-Net** mimarisi kullanıldı.

- **Encoder (Contracting Path):** Semantik bağlamı yakalamak için evrişim katmanları
- **Decoder (Expanding Path):** Skip connection’lar ile uzamsal detayların yüksek çözünürlükte yeniden inşası

---

## 🖼️ Ön İşleme
Model doğruluğunu ve yakınsamayı artırmak için:

- **Yeşil Kanal İzolasyonu:** Damar kontrastının en yüksek olduğu kanal üzerinden işleme  
- **CLAHE (Kontrast Optimizasyonu):** Aydınlatma farklarını azaltıp mikro-damar yapılarını belirginleştirme  
- **Patch-Based Processing:** Sınırlı veri için **64×64** örtüşmeli yamalar (sliding window)

---

## 📈 Performans
Tıbbi segmentasyonda sınıf dengesizliği nedeniyle yalnızca accuracy yanıltıcı olabilir. Bu yüzden ana odak:

- **Dice Coefficient (F1-Score)**
- **Dice Loss** ile doğrudan örtüşme (overlap) maksimize edildi

**Özet Sonuçlar:**
- **%96+ Pixel Accuracy**
- **~%80 Dice Score bandı**  
- **DRIVE** veri setinde kısıtlı görüntü sayısına rağmen **patch-based eğitim + yoğun augmentation** ile güçlü performans

---

## 📁 Klasör Yapısı
```text
├── DRIVE/                  # Orijinal Veri Seti
│   ├── training/           # Eğitim (images + 1st_manual mask)
│   └── test/               # Test verileri
├── Modeller/               # En iyi model ağırlıkları (.keras)
├── Tahminler/              # Model çıktıları / tahmin görselleri
├── main.ipynb              # Ana eğitim + tahmin notebook
└── README.md               # Dokümantasyon


## 📌 Notlar
- Accuracy tek başına segmentasyonda yanıltıcı olabilir (arka plan baskınlığı).
- Dice/IoU gibi metrikler daha anlamlıdır.
- Patch-based yaklaşım, küçük veri setlerinde genelde ciddi fark yaratır.
