# 👁️ DeepRetina: Retinal Damar Segmentasyonu / Retinal Vessel Segmentation

---

## 🇹🇷 


Proje kapsamında, fundus anjiyografi görüntülerinden damar ağını (vascular network) otomatik olarak ayırt edebilen, yüksek hassasiyetli bir derin öğrenme modeli geliştirdim. Çalışmamın temel odağı; diyabetik retinopati ve hipertansiyon gibi sistemik hastalıkların dijital biyobelirteçlerini analiz etmek için güvenilir bir ön aşama segmentasyonu sağlamaktır.

### 🔬 Metodoloji ve Teknik Yaklaşımlar

#### 🛰️ Mimari Tasarım: U-Net
Segmentasyon görevi için, medikal görüntü işlemede üstünlüğü kanıtlanmış olan **U-Net** mimarisini tercih ettim.
* **Encoder (Daralan Yol):** Görüntüdeki semantik bağlamı (context) yakalamak amacıyla evrişimli katmanlar kullandım.
* **Decoder (Genişleyen Yol):** Encoder'dan gelen özellikleri "skip connections" (atlama bağlantıları) üzerinden aktararak, piksellerin uzamsal konumlarını yüksek çözünürlükte yeniden inşa ettim.



#### 🧩 Görüntü İşleme Stratejileri
Modelin yakınsama hızını ve doğruluğunu artırmak adına veri ön işleme aşamasında şu bilimsel adımları izledim:
* **Yeşil Kanal İzolasyonu:** Retina görüntülerinde damar kontrastının en yüksek olduğu spektral aralık yeşil kanal olduğundan, veriyi bu kanal üzerinden işleyerek sinyal-gürültü oranını optimize ettim.
* **Kontrast Optimizasyonu (CLAHE):** Aydınlatma farklılıklarını gidermek ve mikro-vasküler yapıları belirginleştirmek için Kontrast Sınırlı Adaptif Histogram Eşitleme (CLAHE) algoritmasını uyguladım.
* **Patch-Based Processing:** Veri setinin sınırlı yapısını kompanse etmek ve modelin yerel dokuları öğrenmesini sağlamak için görüntüyü 64x64 piksellik örtüşen parçalara (sliding window) bölerek işledim.

### 📈 Performans ve Değerlendirme
Tıbbi görüntü segmentasyonunda "Accuracy" metriği, arka planın (siyah pikseller) baskınlığı nedeniyle yanıltıcı olabilmektedir. Bu nedenle, model başarısını ölçerken **Dice Coefficient (F1-Score)** metriğine odaklandım.
* **Kayıp Fonksiyonu:** Modelimi, segmentasyon isabetini doğrudan maksimize eden **Dice Loss** fonksiyonu ile eğittim.
* **Genelleme Yeteneği:** Veri artırma (Augmentation) teknikleri sayesinde modelin farklı fundus kameralarından gelen görüntülere karşı gürbüz (robust) bir performans sergilemesini sağladım.

### 🎓 Akademik Sonuç
Bu çalışma, derin öğrenme tekniklerinin klinik karar destek sistemlerinde kullanılabilirliğini doğrulamaktadır. Modelim, kılcal damar düzeyindeki detayları yakalayarak manuel segmentasyon ihtiyacını ortadan kaldıran bir performans sergilemektedir.

---

## 🇺🇸 


Within the scope of this project, I developed a high-precision deep learning model capable of automatically segmenting the vascular network from fundus angiography images. The primary focus of my work is to provide a reliable pre-processing segmentation to analyze digital biomarkers of systemic diseases such as diabetic retinopathy and hypertension.

### 🔬 Methodology and Technical Approaches

#### 🛰️ Architectural Design: U-Net
I chose the **U-Net** architecture, which is a gold standard in medical image segmentation, for this task.
* **Encoder (Contracting Path):** Used convolutional layers to capture the semantic context of the image.
* **Decoder (Extensive Path):** Reconstructed spatial positions at high resolution by transferring features from the encoder via "skip connections."

#### 🧩 Image Processing Strategies
I followed these scientific steps during the data preprocessing stage to increase convergence speed and accuracy:
* **Green Channel Isolation:** Since the green channel offers the highest vessel contrast in retinal images, I optimized the signal-to-noise ratio by processing the data through this channel.
* **Contrast Optimization (CLAHE):** Applied the Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm to highlight micro-vascular structures and equalize illumination differences.
* **Patch-Based Processing:** Processed images into 64x64 overlapping patches (sliding window) to compensate for the limited dataset size and help the model learn local textures.



### 📈 Performance and Evaluation
Since the "Accuracy" metric can be misleading in medical image segmentation due to the dominance of background (black) pixels, I focused on the **Dice Coefficient (F1-Score)** metric.
* **Loss Function:** Trained the model using the **Dice Loss** function to directly maximize segmentation overlap.
* **Generalization Ability:** Used **Data Augmentation** techniques to ensure robust performance across images from different fundus cameras.

### 🎓 Academic Conclusion
This work validates the applicability of deep learning techniques in clinical decision support systems. My model demonstrates a performance that eliminates the need for manual segmentation by capturing details at the capillary level.

---

**📜 Citation / Atıf:**
If you use this work in your research or project, please support it by giving a star ⭐. / Eğer bu çalışmayı projelerinizde veya araştırmalarınızda kullanacaksanız, lütfen star ⭐ vererek desteklemeyi unutmayın.
