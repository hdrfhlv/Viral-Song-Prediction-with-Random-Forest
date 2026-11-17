# Hit or Miss? Predicting Viral Songs from Audio Features

Proyek machine learning untuk memprediksi apakah sebuah lagu akan **viral** atau tidak berdasarkan karakteristik audionya, menggunakan algoritma **Random Forest**.

## Latar Belakang

Industri musik terus mencari cara untuk memahami apa yang membuat sebuah lagu menyebar luas. Proyek ini menjawab pertanyaan: *apakah fitur audio dan genre sebuah lagu cukup untuk memprediksi potensi viralnya?*

## Tujuan

- Mengeksplorasi hubungan antara fitur audio, genre, dan status viral
- Membangun model klasifikasi Random Forest untuk memprediksi `viral` (0 / 1)
- Mengevaluasi model dengan metrik yang tepat (precision, recall, F1, AUC)
- Mengidentifikasi fitur yang paling berpengaruh

## Dataset

- **Sumber:** Kaggle (`viral_songs.csv`)
- **Ukuran:** 10.000 lagu, 17 kolom
- **Kualitas data:** tidak ada missing value dan tidak ada duplikat

| Kolom | Deskripsi |
|---|---|
| `track_id`, `track_name`, `artist` | Identitas lagu |
| `genre` | Genre lagu (10 kategori) |
| `release_year` | Tahun rilis (2010 sampai 2025) |
| `duration_sec` | Durasi lagu dalam detik |
| `danceability` | Seberapa cocok lagu untuk menari |
| `energy` | Intensitas dan aktivitas lagu |
| `valence` | Tingkat kepositifan suasana lagu |
| `tempo_bpm` | Tempo (beats per minute) |
| `acousticness` | Tingkat akustik lagu |
| `instrumentalness` | Kecenderungan lagu tanpa vokal |
| `speechiness` | Kehadiran kata-kata yang diucapkan |
| `liveness` | Kehadiran penonton / nuansa live |
| `loudness_db` | Kenyaringan rata-rata (dB) |
| `streams` | Jumlah stream |
| `viral` | **Target**: 1 = viral, 0 = tidak viral |

## Metodologi

1. **Eksplorasi data (EDA):** distribusi fitur, korelasi dengan target, perbandingan antar genre
2. **Preprocessing:**
   - Membuang `track_id`, `track_name`, dan `artist` karena hanya identitas unik
   - Membuang `streams` untuk menghindari **data leakage** (stream adalah akibat dari viral, bukan penyebabnya)
   - Encoding `genre` dengan one-hot encoding (`pd.get_dummies`)
3. **Split data:** 80% latih dan 20% uji dengan `stratify` agar proporsi kelas terjaga
4. **Modeling:** `RandomForestClassifier` dengan evaluasi OOB score
5. **Evaluasi:** accuracy, precision, recall, F1, confusion matrix, ROC-AUC
6. **Tuning:** `RandomizedSearchCV` dengan cross-validation 5-fold
7. **Interpretasi:** feature importance

## Hasil

> Isi bagian ini setelah kamu menyelesaikan eksperimen.

| Model | Accuracy | Precision (viral) | Recall (viral) | F1 (viral) | AUC |
|---|---|---|---|---|---|
| Baseline (kelas mayoritas) | 0.6395 | | | | |
| Decision Tree | 0.6475 | | | | |
| Random Forest (default) | 0.7255 | | | | |
| Random Forest (tuned) | | | | | |

**Temuan utama:**

- _Fitur apa yang paling berpengaruh?_
- _Genre apa yang paling sering viral?_
- _Seberapa baik model menangkap lagu viral?_

## Keterbatasan

- Dataset hanya berisi fitur audio dan genre, tanpa faktor eksternal seperti promosi, tren media sosial, atau popularitas artis
- Model memprediksi korelasi, bukan sebab-akibat
- Performa diukur pada satu split data; hasil bisa sedikit berbeda dengan split lain

## Teknologi

- Python 3
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- joblib

## Struktur Proyek

```
viral-song-prediction/
├── data/
│   └── viral_songs.csv
├── viral_song_prediction.ipynb
├── requirements.txt
└── README.md
```

## Cara Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/<username>/viral-song-prediction.git
cd viral-song-prediction

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan notebook
jupyter notebook viral_song_prediction.ipynb
```

Isi `requirements.txt`:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
jupyter
```

## Pengembangan Selanjutnya

- Mencoba model lain (Gradient Boosting, XGBoost, LightGBM)
- Menyesuaikan threshold prediksi untuk meningkatkan recall
- Menggunakan SHAP untuk interpretasi model yang lebih mendalam
- Membuat aplikasi sederhana (Streamlit) untuk memprediksi lagu baru

## Penulis

**<Nama kamu>**
[GitHub](https://github.com/<username>) | [LinkedIn](https://linkedin.com/in/<username>)

## Lisensi

Proyek ini dibuat untuk tujuan pembelajaran. Dataset mengikuti lisensi dari sumber aslinya di Kaggle.
