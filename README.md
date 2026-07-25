# Indonesian License Plate OCR using Vision-Language Model (LM Studio + Moondream2)

Proyek ini melakukan **Optical Character Recognition (OCR)** pada pelat nomor kendaraan Indonesia menggunakan **Vision-Language Model (VLM)** yang dijalankan secara lokal melalui **LM Studio**, kemudian mengevaluasi hasilnya dengan metrik **Character Error Rate (CER)**.

## 1. Ringkasan Proyek

| Item | Keterangan |
|---|---|
| Model VLM | `moondream-2b-2025-04-14` (via LM Studio, API kompatibel OpenAI) |
| Dataset | Indonesian License Plate Recognition Dataset (format YOLO: images + labels) |
| Total gambar diuji | 197 gambar plat nomor |
| Metrik evaluasi | Character Error Rate (CER) berbasis Levenshtein distance |
| Rata-rata CER | **0.3689** (semakin kecil semakin baik) |
| Exact match (CER = 0) | 60 gambar — **Accuracy (exact): 30.5%** |
| Berhasil (CER ≤ 0.5) | 131 gambar (**66.5%**) |
| Gagal (CER > 0.5) | 66 gambar (**33.5%**) |
| Output | `hasil_evaluasi_ocr_lmstudio.csv` |

## 2. Struktur Folder

```
RE604_Muzammil-AAS/
├── ocr_vlm.py                                  # skrip utama inferensi + evaluasi
├── hasil_evaluasi_ocr_lmstudio.csv           # hasil OCR + skor CER per gambar
├── README.md
└── Indonesian License Plate Recognition Dataset/
    ├── classes.names                        # daftar karakter (A-Z, 0-9, dst)
    ├── images/
    │   └── test/                            # gambar pelat nomor (.jpg)
    └── labels/
        └── test/                            # label YOLO (.txt) sebagai ground truth
```

## 3. Prasyarat (Requirements)

- Python 3.9 atau lebih baru
- [LM Studio](https://lmstudio.ai/) terinstal di komputer lokal
- Model VLM `moondream-2b-2025-04-14` (atau model vision lain yang kompatibel) sudah diunduh di LM Studio
- (Opsional, untuk fallback) [EasyOCR](https://github.com/JaidedAI/EasyOCR)

### Instalasi dependensi Python

```bash
pip install requests
pip install easyocr
```

## 4. Menyiapkan LM Studio (Server VLM Lokal)

1. Buka aplikasi **LM Studio**.
2. Unduh model **`moondream-2b-2025-04-14`** (atau model vision-language lain yang tersedia) melalui tab *Search*.
3. Masuk ke tab **Local Server** (ikon panah dua arah), pilih model yang telah diunduh, lalu klik **Start Server**.
4. Pastikan server berjalan di endpoint berikut. Ini adalah URL yang dipanggil skrip Python melalui variabel `LMSTUDIO_BASE_URL`:

   ```
   http://127.0.0.1:1234/v1
   ```

5. Uji koneksi server secara manual (opsional). Jika server aktif, akan muncul daftar model yang tersedia dalam format JSON:

   ```bash
   curl http://127.0.0.1:1234/v1/models
   ```

## 5. Menyiapkan Dataset

1. Unduh **Indonesian License Plate Recognition Dataset** dan letakkan pada folder `Indonesian License Plate Recognition Dataset/` di root proyek, mengikuti struktur pada Bagian 2.
2. Pastikan berkas `classes.names` berisi daftar karakter sesuai indeks kelas pada label YOLO.
3. Setiap gambar pada `images/test/` harus memiliki pasangan label pada `labels/test/` dengan nama file yang sama (`.txt`), berisi anotasi karakter dalam format YOLO (`class_id x_center y_center width height`).

## 6. Konfigurasi Skrip

Konfigurasi utama berada di bagian atas `main.py`, sesuaikan bila perlu:

```python
DATASET_BASE = "Indonesian License Plate Recognition Dataset"
IMAGES_DIR   = os.path.join(DATASET_BASE, "images", "test")
LABELS_DIR   = os.path.join(DATASET_BASE, "labels", "test")
CLASSES_FILE = os.path.join(DATASET_BASE, "classes.names")
OUTPUT_CSV   = "hasil_evaluasi_ocr_lmstudio.csv"

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
MODEL_NAME = "moondream-2b-2025-04-14"
PROMPT = "Extract the license plate number from this image. Output only the alphanumeric characters, no spaces, no punctuation."
```

## 7. Menjalankan Program

1. Pastikan server LM Studio sudah aktif (Bagian 4).
2. Jalankan skrip dari root proyek:

   ```bash
   python ocr_vlm.py
   ```

3. Saat dijalankan, program akan melakukan hal-hal berikut secara berurutan:
   - Mengecek koneksi ke server LM Studio (`GET /v1/models`).
   - Menguji satu gambar contoh terlebih dahulu untuk memverifikasi respons model.
   - Menampilkan prompt konfirmasi berikut untuk menentukan apakah menggunakan EasyOCR sebagai fallback:

     ```
     Apakah ingin menggunakan EasyOCR sebagai fallback? (y/n):
     ```

     Jawab `n` untuk menjalankan inferensi murni menggunakan VLM (LM Studio), atau `y` untuk menggunakan EasyOCR sebagai metode utama/fallback.
   - Melakukan iterasi ke seluruh gambar pada `images/test/`, memanggil VLM melalui endpoint `/chat/completions`, mengekstrak teks plat nomor dari respons, lalu menghitung CER terhadap ground truth dari label YOLO.
4. Setelah selesai, hasil disimpan ke `hasil_evaluasi_ocr_lmstudio.csv` dan rata-rata CER ditampilkan di terminal.

## 8. Format Output (`hasil_evaluasi_ocr_lmstudio.csv`)

| Kolom | Deskripsi |
|---|---|
| `image` | Nama file gambar |
| `ground_truth` | Teks plat nomor sebenarnya (dari label YOLO) |
| `prediction` | Teks hasil prediksi VLM/EasyOCR |
| `CER_score` | Character Error Rate, dihitung sebagai `edit_distance(gt, pred) / len(gt)` |

Contoh baris:
```
image,ground_truth,prediction,CER_score
test001_2.jpg,B2407UZO,B2407UZO,0.0
test006_1.jpg,T1329KC,ID4037,1.0
```

## 9. Metodologi Perhitungan CER

CER dihitung menggunakan **jarak Levenshtein** (edit distance) antara string ground truth dan prediksi, dibagi dengan panjang ground truth:

```
CER = EditDistance(ground_truth, prediction) / len(ground_truth)
```

- `CER = 0.0` → prediksi identik dengan ground truth (sempurna).
- `CER` mendekati/lebih dari `1.0` → prediksi hampir/seluruhnya salah (bisa lebih dari 1.0 jika hasil prediksi lebih panjang dari ground truth dan banyak yang salah, misalnya kasus `test035_2.jpg` dengan CER 1.1429).

## 10. Ringkasan Hasil Evaluasi

Output akhir program di terminal:

```
Selesai.
Rata-rata CER: 0.3689
Hasil di: hasil_evaluasi_ocr_lmstudio.csv
```

### Ringkasan Evaluasi OCR

| Metrik | Nilai |
|---|---|
| Total gambar | 197 |
| Rata-rata CER | 0.3689 |
| Exact match (CER = 0) | 60 |
| Accuracy (exact) | 30.5% |
| Berhasil (CER ≤ 0.5) | 131 (66.5%) |
| Gagal (CER > 0.5) | 66 (33.5%) |

### 5 Contoh Sukses (CER Terendah)

| No | Gambar | Ground Truth | Prediksi | CER |
|---|---|---|---|---|
| 1 | test001_2.jpg | B2407UZO | B2407UZO | 0.0000 |
| 2 | test001_3.jpg | B2842PKM | B2842PKM | 0.0000 |
| 3 | test003_1.jpg | B2634UZF | B2634UZF | 0.0000 |
| 4 | test003_2.jpg | B1995JVK | B1995JVK | 0.0000 |
| 5 | test007_1.jpg | AD8865EE | AD8865EE | 0.0000 |

Kelima contoh di atas menunjukkan prediksi VLM identik 100% dengan ground truth. Kesamaan pola pada gambar-gambar ini umumnya pencahayaan merata, sudut pengambilan gambar tegak lurus terhadap pelat, dan karakter tidak terhalang objek lain.

### 5 Contoh Gagal (CER Tertinggi)

| No | Gambar | Ground Truth | Prediksi | CER |
|---|---|---|---|---|
| 1 | test035_2.jpg | L5247GI | 5227000228 | 1.1429 |
| 2 | test097_2.jpg | B6354SVL | VL0225 | 1.0000 |
| 3 | test090_1.jpg | T9314E | E0I | 1.0000 |
| 4 | test087_4.jpg | 4039NHR | HR0724 | 1.0000 |
| 5 | test084_1.jpg | B1158PYP | YP0616 | 1.0000 |

Pada kasus-kasus ini, prediksi VLM nyaris tidak memiliki kemiripan karakter dengan ground truth (CER = 1.0, bahkan lebih dari 1.0 pada `test035_2.jpg` karena panjang prediksi melebihi panjang ground truth). Pola kegagalan yang teramati:

- **Hanya membaca sebagian pelat**, misalnya `test097_2.jpg` (`B6354SVL` → `VL0225`) dan `test084_1.jpg` (`B1158PYP` → `YP0616`), model tampak hanya menangkap 2 karakter terakhir lalu menambahkan angka acak.
- **Salah total membaca karakter**, seperti `test090_1.jpg` (`T9314E` → `E0I`) dan `test087_4.jpg` (`4039NHR` → `HR0724`), kemungkinan akibat gambar buram, resolusi rendah, atau sudut pengambilan yang ekstrem.
- **Prediksi berlebihan/halusinasi**, seperti `test035_2.jpg` (`L5247GI` → `5227000228`), di mana model menghasilkan lebih banyak digit daripada yang sebenarnya ada pada pelat.

### Interpretasi Umum

- Sekitar **1 dari 3 gambar (30.5%)** berhasil dibaca sempurna tanpa kesalahan karakter sama sekali.
- Dengan ambang toleransi kesalahan CER ≤ 0.5, **66.5% gambar** masih dianggap "berhasil" (sebagian besar karakter terbaca benar).
- **33.5% gambar** dikategorikan gagal (CER > 0.5), didominasi oleh kasus pencahayaan buruk, sudut kamera miring, karakter tertutup/blur, atau model yang hanya menangkap sebagian pelat.
- Kesalahan yang sering muncul pada kasus ringan biasanya berupa kekeliruan antar karakter yang bentuknya mirip (`B` vs `8`, `O` vs `0`, `S` vs `5`).

## 11. Troubleshooting

| Masalah | Solusi |
|---|---|
| `Tidak dapat terhubung ke LMStudio` | Pastikan aplikasi LM Studio terbuka dan tombol **Start Server** pada tab Local Server sudah aktif. |
| `Folder ... tidak ditemukan` | Periksa kembali path `DATASET_BASE` dan pastikan struktur folder dataset sesuai Bagian 2. |
| Respons VLM kosong / bukan pelat nomor | Coba turunkan `max_tokens`, sesuaikan `PROMPT`, atau gunakan mode fallback EasyOCR (`y` saat ditanya). |
| CER tinggi secara konsisten | Periksa kualitas gambar input, resolusi, dan pencahayaan; pertimbangkan model VLM lain yang lebih besar. |

## 12. Lisensi & Atribusi

Dataset yang digunakan adalah **Indonesian License Plate Recognition Dataset**, dan model VLM yang digunakan adalah **Moondream2**, dijalankan secara lokal melalui **LM Studio**. Proyek ini dibuat untuk lebih memahami pengaplikasian VLM serta tugas akhir semester.
