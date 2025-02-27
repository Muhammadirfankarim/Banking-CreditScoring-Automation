# Pipeline Machine Learning End-to-End untuk Bank Marketing dengan Jenkins

Proyek ini mengimplementasikan pipeline machine learning end-to-end untuk memprediksi apakah klien bank akan berlangganan deposito berjangka (variabel y) menggunakan dataset Bank Marketing. Pipeline dijalankan secara otomatis menggunakan Jenkins untuk otomatisasi dan monitoring.

## Gambaran Proyek

Proyek ini mencakup komponen-komponen berikut:

1. **Exploratory Data Analysis (EDA)**: Analisis dan visualisasi data untuk menghasilkan insight
2. **Preprocessing Data**: Pembersihan dan transformasi data untuk persiapan pemodelan
3. **Pemodelan Machine Learning**: Melatih dan mengevaluasi beberapa model untuk memilih yang terbaik
4. **Deployment Model**: Men-deploy model terbaik ke webserver untuk monitoring dan prediksi
5. **Otomatisasi dengan Jenkins**: Pipeline CI/CD untuk otomatisasi seluruh proses

## Struktur Proyek

```
bank_credit_scoring_automation_jenkins/
├── bank.csv                  # Data Bank Marketing
├── bank_ml_pipeline.py       # Kode utama untuk pipeline ML
├── app.py                    # Webserver Flask untuk monitoring dan prediksi
├── Jenkinsfile               # Konfigurasi pipeline Jenkins
├── templates/                # Template HTML untuk webserver
│   └── index.html            # Dashboard monitoring
├── models/                   # Direktori untuk menyimpan model terlatih
├── reports/                  # Direktori untuk laporan dan metrik
├── visualizations/           # Direktori untuk visualisasi dan plot
├── data/                     # Direktori untuk data hasil preprocessing
└── predictions/              # Direktori untuk menyimpan prediksi baru
```

## Dataset

Dataset Bank Marketing bersumber dari UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/222/bank+marketing

Dataset berisi data kampanye pemasaran bank, dengan informasi demografis dan riwayat transaksi pelanggan. Target prediksi adalah apakah pelanggan akan berlangganan deposito berjangka ('yes') atau tidak ('no').

## Fitur-fitur

1. **age**: Usia pelanggan (numerik)
2. **job**: Jenis pekerjaan (kategorikal)
3. **marital**: Status pernikahan (kategorikal)
4. **education**: Tingkat pendidikan (kategorikal)
5. **default**: Apakah memiliki kredit default (kategorikal)
6. **balance**: Saldo rata-rata tahunan (numerik)
7. **housing**: Apakah memiliki pinjaman rumah (kategorikal)
8. **loan**: Apakah memiliki pinjaman pribadi (kategorikal)
9. **contact**: Jenis kontak komunikasi (kategorikal)
10. **day**: Hari terakhir dihubungi (numerik)
11. **month**: Bulan terakhir dihubungi (kategorikal)
12. **duration**: Durasi kontak terakhir dalam detik (numerik)
13. **campaign**: Jumlah kontak selama kampanye ini (numerik)
14. **pdays**: Jumlah hari setelah pelanggan terakhir dihubungi (numerik)
15. **previous**: Jumlah kontak sebelum kampanye ini (numerik)
16. **poutcome**: Hasil kampanye pemasaran sebelumnya (kategorikal)

**Target**:
- **y**: Apakah pelanggan berlangganan deposito berjangka ('yes' atau 'no')

## Pipeline Machine Learning

### 1. Exploratory Data Analysis (EDA)

Tahap ini mencakup:
- Analisis statistik deskriptif
- Visualisasi distribusi fitur
- Analisis korelasi antar fitur
- Analisis hubungan fitur dengan target
- Pembuatan insight dan rekomendasi

### 2. Preprocessing Data

Tahap ini mencakup:
- Penanganan missing values
- Encoding variabel kategorikal
- Penskalaan fitur numerik
- Pembagian data menjadi training dan testing set

### 3. Model Training

Beberapa model machine learning dilatih dan dievaluasi:
- Logistic Regression
- Random Forest
- Gradient Boosting

Model terbaik dipilih berdasarkan skor ROC-AUC pada validation set.

### 4. Evaluasi Model

Model dievaluasi menggunakan metrik:
- ROC-AUC
- Precision-Recall AUC
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix

### 5. Deploy dan Monitoring

Model di-deploy ke webserver Flask dengan fitur:
- Endpoint API untuk prediksi
- Dashboard monitoring performa model
- Analisis data drift
- Retraining model

## Penggunaan Jenkins

Pipeline Jenkins mengotomatisasi seluruh proses machine learning:

1. **Setup Environment**: Mempersiapkan environment dan dependensi
2. **Data Validation**: Memvalidasi kualitas dan struktur data
3. **EDA**: Menjalankan analisis data dan menghasilkan insight
4. **Preprocessing**: Memproses data untuk model
5. **Model Training**: Melatih dan mengevaluasi model
6. **Deploy Webserver**: Men-deploy model ke webserver monitoring
7. **Functional Testing**: Menguji fungsionalitas API dan webserver

## Cara Menjalankan

### Prasyarat

- Python 3.7+
- Jenkins
- Packages Python: pandas, numpy, scikit-learn, matplotlib, seaborn, flask

### Langkah-langkah

1. Clone repositori:
```
git clone <repository-url>
cd bank_credit_scoring_automation_jenkins
```

2. Konfigurasi Jenkins:
   - Buat job baru di Jenkins
   - Konfigurasi pipeline menggunakan Jenkinsfile pada repositori
   - Mulai build pipeline

3. Akses dashboard monitoring:
   - Buka browser dan akses `http://localhost:5000` setelah pipeline selesai

## Kontribusi

Kontribusi terhadap proyek ini sangat diterima. Silakan fork repositori dan buat pull request.

## Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file LICENSE untuk detail selengkapnya.

## Kontak

Jika Anda memiliki pertanyaan, silakan hubungi Muhammad Irfan Karim - [karimirfan51@gmail.com] 