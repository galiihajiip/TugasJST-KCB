# LAPORAN TUGAS JARINGAN SARAF TIRUAN (JST)

**KELOMPOK 4**

**Dosen Pengampu:** Yisti Vita Via, S.ST. M.Kom.

**Disusun Oleh:**
- Galih Aji Pangestu (24081010123)
- Arij Fiddin Al Wafa (24081010304)
- Badii Ul Choir Al Irsyad (24081010196)
- Gavin Rizqi Allfathi (24081010313)
- Dhamar Rizky Ananda (24081010201)

**MATA KULIAH KECERDASAN BUATAN**  
**PROGRAM STUDI INFORMATIKA FAKULTAS ILMU KOMPUTER**  
**UNIVERSITAS PEMBANGUNAN NASIONAL "VETERAN" JAWA TIMUR**  
**2026**

---

## BAB I: PENDAHULUAN

### 1.1 Latar Belakang
Jaringan Syaraf Tiruan (JST) adalah model komputasi yang terinspirasi dari cara kerja otak manusia, terdiri dari neuron-neuron buatan yang saling terhubung, dan mampu belajar dari data untuk mengenali pola serta membuat prediksi. Konsep ini pertama kali diperkenalkan oleh Warren McCulloch dan Walter Pitts pada tahun 1943.

Dalam tugas ini, kami mengimplementasikan JST sederhana untuk menyelesaikan permasalahan logika dasar menggunakan dua jenis arsitektur:
1. **Single Layer Perceptron** untuk kasus AND dan OR (linear separable)
2. **Multilayer Perceptron (MLP) dengan Backpropagation** untuk kasus XOR (non-linear separable)

Implementasi dilakukan mengikuti **skenario percobaan** yang ditetapkan dosen, mencakup variasi bobot awal (ditentukan vs. acak) dan variasi learning rate.

### 1.2 Tujuan
1. Memahami dan mengimplementasikan konsep dasar JST
2. Membangun Single Layer Perceptron untuk kasus AND dan OR
3. Membangun MLP Backpropagation untuk kasus XOR
4. Menganalisis pengaruh inisialisasi bobot awal terhadap konvergensi
5. Menganalisis pengaruh learning rate terhadap kecepatan konvergensi
6. Menganalisis pengaruh jumlah hidden node pada MLP XOR
7. Menjelaskan mengapa XOR tidak dapat diselesaikan single layer perceptron

### 1.3 Dataset
Dataset yang digunakan adalah tabel kebenaran logika AND, OR, dan XOR dengan 2 input (x1, x2) dan 1 output (target).

| x1 | x2 | AND | OR | XOR |
|:--:|:--:|:---:|:--:|:---:|
| 0  | 0  |  0  |  0 |  0  |
| 0  | 1  |  0  |  1 |  1  |
| 1  | 0  |  0  |  1 |  1  |
| 1  | 1  |  1  |  1 |  0  |

---

## BAB II: KONSEP DASAR JST

### 2.1 Definisi JST
Jaringan Syaraf Tiruan (JST) adalah model komputasi yang terinspirasi dari cara kerja otak manusia. JST terdiri dari neuron-neuron buatan yang saling terhubung dan mampu belajar dari data untuk mengenali pola serta membuat prediksi.

### 2.2 Komponen JST
**a. Input (Masukan)**
Data yang dimasukkan ke dalam jaringan. Dalam kasus ini: x1 dan x2 (nilai 0 atau 1).

**b. Bobot (Weight) dan Bias**
Bobot (w) menentukan kekuatan koneksi antar neuron. Bias (b) adalah nilai tambahan yang mempengaruhi threshold aktivasi. Keduanya diperbarui selama proses training.

**c. Fungsi Aktivasi**
- **Single Layer (AND/OR):** Step Function → `f(net) = 1` jika `net >= 0`, `f(net) = 0` jika `net < 0`
- **MLP (XOR):** Sigmoid Function → `f(net) = 1 / (1 + e^(-net))`, turunan dipakai untuk backpropagation

**d. Lapisan (Layer)**
1. **Input Layer:** Menerima data masukan (x1, x2)
2. **Hidden Layer:** Lapisan tersembunyi (khusus MLP untuk XOR, jumlah node bervariasi)
3. **Output Layer:** Menghasilkan prediksi/keputusan

### 2.3 Rumus Perceptron (Single Layer)
Rumus perhitungan net input:
```
net = w1 × x1 + w2 × x2 + b
```

Rumus update bobot (Perceptron Rule):
```
w_baru = w_lama + α × (t - y) × xi
b_baru = b_lama + α × (t - y)
```
*(Keterangan: α = learning rate, t = target, y = output JST, xi = input ke-i)*

### 2.4 Rumus MLP Backpropagation (XOR)
**Forward Pass:**
```
net_h = X · W1 + b1         (input ke hidden)
H     = sigmoid(net_h)       (output hidden)
net_o = H · W2 + b2          (input ke output)
Y     = sigmoid(net_o)        (output akhir)
```

**Backward Pass (update bobot):**
```
δ_output = (t - Y) × sigmoid'(net_o)
δ_hidden = (δ_output · W2ᵀ) × sigmoid'(net_h)

W2 += α × Hᵀ · δ_output
W1 += α × Xᵀ · δ_hidden
```

### 2.5 Tahapan Training JST
1. Inisialisasi bobot dan bias (nilai awal, bisa ditentukan atau acak)
2. Forward pass: hitung net dan output
3. Hitung error / MSE
4. Jika belum konvergen, lakukan backward pass dan update bobot
5. Ulangi (epoch) sampai konvergen atau mencapai epoch maksimal

---

## BAB III: SKENARIO PERCOBAAN

Skenario percobaan ditetapkan oleh dosen untuk menguji pengaruh berbagai parameter terhadap proses training JST.

### 3.1 Skenario AND & OR (Single Layer Perceptron)

| # | Skenario | Bobot Awal | Learning Rate |
|:-:|:---------|:-----------|:-------------:|
| 1 | Bobot ditentukan, dibandingkan dg Excel | AND: w1=0.2, w2=0.2, b=-0.1 / OR: w1=0.0, w2=0.0, b=-0.5 | 0.1 |
| 2 | Bobot random -0.5 s.d 0.5 | Acak uniform [-0.5, 0.5] | 0.1 |
| 3 | Bobot random -1.0 s.d 1.0 | Acak uniform [-1.0, 1.0] | 0.1 |
| 4 | Bobot ditentukan, multi learning rate | AND: w1=0.2, w2=0.2, b=-0.1 / OR: w1=0.0, w2=0.0, b=-0.5 | 0.01, 0.05, 0.1, 0.5 |

### 3.2 Skenario XOR (MLP Backpropagation)

| # | Skenario | Bobot Awal | Learning Rate | Hidden Node |
|:-:|:---------|:-----------|:-------------:|:-----------:|
| 1 | Bobot ditentukan, dibandingkan dg Excel | Seed tetap (seed=42), range ±0.5 | 0.1 | 2 |
| 2 | Bobot random -0.5 s.d 0.5 | Acak uniform [-0.5, 0.5] | 0.1 | 2 |
| 3 | Bobot random -1.0 s.d 1.0 | Acak uniform [-1.0, 1.0] | 0.1 | 2 |
| 4 | Bobot ditentukan, multi learning rate | Seed tetap (seed=42), range ±0.5 | 0.01, 0.05, 0.1, 0.5 | 2 |
| 5 | Random ±0.5, variasi hidden node | Acak uniform [-0.5, 0.5] | 0.1 | **2, 3, 5** |

---

## BAB IV: DESAIN ARSITEKTUR MODEL

### 4.1 Single Layer Perceptron (AND & OR)
Arsitektur: **2 input → 1 output** (tanpa hidden layer)

```
x1 ──[w1]──┐
            ├──[Σ + b]──[Step]── y
x2 ──[w2]──┘
```

| Parameter | AND | OR |
|:----------|:---:|:--:|
| Bobot awal w1 | 0.2 | 0.0 |
| Bobot awal w2 | 0.2 | 0.0 |
| Bias awal b | -0.1 | -0.5 |
| Fungsi aktivasi | Step | Step |
| Max epoch | 1000 | 1000 |

### 4.2 Multilayer Perceptron (XOR)
Arsitektur dasar: **2 input → 2 hidden → 1 output** (2-2-1)

```
x1 ──[W1]──┐             ┌──[W2]──┐
            ├──[H layer]─┤        ├──[Σ + b]──[Sigmoid]── y
x2 ──[W1]──┘             └──[W2]──┘
```

| Parameter | Nilai |
|:----------|:-----:|
| Fungsi aktivasi | Sigmoid |
| Konvergen jika | MSE < 0.01 |
| Max epoch | 10.000 |
| Arsitektur yang diuji | 2-2-1, 2-3-1, 2-5-1 |

---

## BAB V: IMPLEMENTASI

### 5.1 Struktur Program

```
KCB/
├── main.py          # Program terminal (semua skenario)
├── app.py           # Dashboard interaktif Streamlit
├── README.md        # Laporan ini
└── grafik_*.png     # Grafik konvergensi (otomatis dihasilkan)
```

### 5.2 Cara Menjalankan

**Terminal (semua skenario sekaligus):**
```bash
python main.py
```

**interaktif Streamlit:**
```bash
streamlit run app.py
```

### 5.3 Dependensi
```
numpy
matplotlib
streamlit
pandas
```

---

## BAB VI: HASIL PERCOBAAN

### 6.1 Hasil AND – Single Layer Perceptron

**Skenario 1 – Bobot Ditentukan (w1=0.2, w2=0.2, b=-0.1), LR=0.1**

| No | x1 | x2 | t | net | y | error | Status |
|:--:|:--:|:--:|:-:|:---:|:-:|:-----:|:------:|
| 1 | 0 | 0 | 0 | -0.10 | 0 | 0 | Benar |
| 2 | 0 | 1 | 0 | 0.10 | 1 | -1 | Salah |
| 3 | 1 | 0 | 0 | 0.10 | 1 | -1 | Salah |
| 4 | 1 | 1 | 1 | 0.30 | 1 | 0 | Benar |

Bobot Final: `w1=0.20, w2=0.20, b=-0.20` | **Konvergen Epoch 2** | Akurasi 100%

**Skenario 4 – Perbandingan Multi Learning Rate (bobot tetap)**

| LR | Epoch Konvergen | W1 Final | W2 Final | b Final |
|:--:|:---------------:|:--------:|:--------:|:-------:|
| 0.01 | 5 | 0.1700 | 0.1600 | -0.1700 |
| 0.05 | 2 | 0.1500 | 0.1500 | -0.2000 |
| 0.1  | 2 | 0.2000 | 0.2000 | -0.2000 |
| 0.5  | 8 | 1.2000 | 0.7000 | -1.6000 |

> LR terlalu kecil (0.01) butuh lebih banyak epoch. LR terlalu besar (0.5) menyebabkan bobot melompat jauh, namun tetap konvergen karena AND bersifat linear separable.

---

### 6.2 Hasil OR – Single Layer Perceptron

**Skenario 1 – Bobot Ditentukan (w1=0.0, w2=0.0, b=-0.5), LR=0.1**

Bobot Final: `w1=0.20, w2=0.30, b=-0.10` | **Konvergen Epoch 3** | Akurasi 100%

**Skenario 4 – Perbandingan Multi Learning Rate (bobot tetap)**

| LR | Epoch Konvergen | W1 Final | W2 Final | b Final |
|:--:|:---------------:|:--------:|:--------:|:-------:|
| 0.01 | 13 | 0.1900 | 0.1900 | -0.1900 |
| 0.05 | 4  | 0.2000 | 0.2000 | -0.1500 |
| 0.1  | 3  | 0.2000 | 0.3000 | -0.1000 |
| 0.5  | 4  | 0.5000 | 0.5000 | -0.5000 |

---

### 6.3 Hasil XOR – MLP Backpropagation

**Skenario 1 – Bobot Ditentukan (seed=42, ±0.5), LR=0.1, Hidden=2**

Status: **Belum Konvergen** (>10.000 epoch) | Akurasi: 75%

> Ini menunjukkan bahwa inisialisasi bobot awal sangat berpengaruh pada XOR. Seed 42 dengan 2 hidden node dan LR=0.1 terjebak di local minimum.

**Skenario 2 – Random ±0.5, LR=0.1, 3 Run**

| Run | Seed | Epoch Konvergen | MSE Akhir | Akurasi |
|:---:|:----:|:---------------:|:---------:|:-------:|
| 1 | 3  | >10.000 | 0.249981 | 50% |
| 2 | 20 | 9.032   | 0.009995 | 100% |
| 3 | 37 | 9.324   | 0.009999 | 100% |

> Bobot awal acak dengan seed berbeda menghasilkan konvergensi yang berbeda-beda — ini menunjukkan XOR sangat sensitif terhadap inisialisasi bobot.

**Skenario 4 – Bobot Tetap (seed=42), Multi LR**

| LR | Epoch Konvergen | Akurasi | MSE Akhir |
|:--:|:---------------:|:-------:|:---------:|
| 0.01 | >10.000 | 50%  | 0.250041 |
| 0.05 | >10.000 | 50%  | 0.249982 |
| 0.1  | >10.000 | 75%  | 0.246318 |
| 0.5  | 2.888   | 100% | 0.009995 |

> Untuk XOR dengan bobot awal ini, hanya LR=0.5 yang berhasil konvergen. LR kecil tidak cukup kuat untuk keluar dari dataran error (saddle point/plateau) di sekitar MSE=0.25.

**Skenario 5 – Perbandingan Hidden Node 2 vs 3 vs 5 (Random ±0.5, LR=0.1)**

| Arsitektur | Hidden Node | Epoch Konvergen | Akurasi | MSE Akhir |
|:----------:|:-----------:|:---------------:|:-------:|:---------:|
| 2-2-1 | 2 | >10.000 | 75%  | 0.246318 |
| 2-3-1 | 3 | 5.635   | 100% | 0.009999 |
| 2-5-1 | 5 | 7.888   | 100% | 0.009999 |

**Forward pass hasil akhir MLP (arsitektur 2-3-1, akurasi 100%):**

| No | x1 | x2 | Target | Output (sigmoid) | Output (biner) | Status |
|:--:|:--:|:--:|:------:|:----------------:|:--------------:|:------:|
| 1 | 0 | 0 | 0 | ~0.10 | 0 | Benar |
| 2 | 0 | 1 | 1 | ~0.89 | 1 | Benar |
| 3 | 1 | 0 | 1 | ~0.90 | 1 | Benar |
| 4 | 1 | 1 | 0 | ~0.09 | 0 | Benar |

---

## BAB VII: ANALISIS HASIL

### 7.1 Pengaruh Inisialisasi Bobot Awal

**AND & OR (Single Layer):**
- Bobot awal yang berbeda (ditentukan, random ±0.5, random ±1.0) tetap menghasilkan konvergensi 100% karena masalah AND/OR bersifat *linear separable*
- Perbedaan hanya pada jumlah epoch yang dibutuhkan

**XOR (MLP):**
- Inisialisasi bobot sangat menentukan apakah model berhasil konvergen atau tidak
- Beberapa kombinasi bobot awal + learning rate menyebabkan model terjebak di plateau (MSE ≈ 0.25) yang merupakan *local minimum*
- Ini adalah karakteristik masalah non-linear separable yang dioptimasi dengan gradient descent

### 7.2 Pengaruh Learning Rate

| Efek | LR Kecil (0.01) | LR Sedang (0.1) | LR Besar (0.5) |
|:-----|:---------------:|:---------------:|:--------------:|
| Kecepatan | Lambat | Sedang | Cepat (atau gagal) |
| Stabilitas | Stabil | Stabil | Dapat berosilasi |
| Risiko | Terlalu lambat konvergen | Seimbang | Melewati minimum |
| AND/OR | Konvergen (lebih banyak epoch) | Konvergen | Konvergen |
| XOR (seed=42) | Tidak konvergen | Tidak konvergen | Konvergen |

> **Kesimpulan:** Tidak ada satu learning rate terbaik untuk semua kasus. Untuk XOR dengan inisialisasi tertentu, LR besar justru diperlukan untuk keluar dari local minimum.

### 7.3 Pengaruh Jumlah Hidden Node (XOR)

| Hidden Node | Kemampuan | Epoch (seed=42) | Keterangan |
|:-----------:|:---------:|:---------------:|:-----------|
| 2 | Terbatas | >10.000 (gagal) | Representasi terlalu sederhana untuk bobot ini |
| 3 | Cukup | 5.635 | Konvergen lebih cepat, representasi lebih kaya |
| 5 | Lebih dari cukup | 7.888 | Konvergen, tapi lebih lambat dari 3 (overfitting capacity) |

> **Kesimpulan:** Lebih banyak hidden node tidak selalu lebih baik. Node=3 lebih optimal dari node=5 untuk kasus XOR ini karena kompleksitas yang tepat tanpa overhead berlebih.

### 7.4 Perbandingan AND, OR, dan XOR

| Kasus | Model | Sifat | Konvergen? | Akurasi |
|:------|:------|:------|:----------:|:-------:|
| AND | Single Layer | Linear separable | Ya (cepat) | 100% |
| OR  | Single Layer | Linear separable | Ya (cepat) | 100% |
| XOR | Single Layer | Non-linear | Tidak | <100% |
| XOR | MLP 2-2-1 | Non-linear | Bergantung seed/LR | 50–100% |
| XOR | MLP 2-3-1 | Non-linear | Ya | 100% |
| XOR | MLP 2-5-1 | Non-linear | Ya | 100% |

### 7.5 Mengapa XOR Tidak Bisa Diselesaikan Single Layer
1. **XOR bersifat NON-LINEARLY SEPARABLE.** Tidak ada garis lurus yang bisa memisahkan titik XOR=1 dari XOR=0 di ruang 2D.
2. **Single Layer hanya membentuk batas keputusan linier.** Perceptron Rule hanya menghasilkan hyperplane, bukan kurva.
3. **MLP mengatasi ini** dengan hidden layer yang mentransformasi ruang input ke representasi baru yang linear separable sebelum diklasifikasikan di output layer.

---

## BAB VIII: KESIMPULAN

1. **AND dan OR** berhasil diselesaikan Single Layer Perceptron dengan akurasi 100% di semua skenario bobot awal maupun variasi learning rate, karena keduanya bersifat *linear separable*.

2. **XOR tidak bisa** diselesaikan Single Layer Perceptron karena bersifat *non-linearly separable*. Dibutuhkan MLP dengan hidden layer.

3. **Inisialisasi bobot** tidak terlalu berpengaruh pada AND/OR, namun sangat kritis pada XOR — bobot awal yang buruk dapat membuat MLP terjebak di *local minimum*.

4. **Learning rate** berpengaruh pada kecepatan dan keberhasilan konvergensi. Untuk XOR dengan inisialisasi tertentu, LR besar (0.5) lebih efektif daripada LR kecil yang terjebak di plateau MSE ≈ 0.25.

5. **Jumlah hidden node** berpengaruh signifikan pada XOR. Node=2 sering gagal, node=3 optimal, node=5 tetap konvergen namun tidak lebih cepat dari node=3 untuk kasus ini.

6. **Temuan menarik dari skenario 5:** Hidden node=3 lebih cepat konvergen (5.635 epoch) dibanding hidden node=5 (7.888 epoch), menunjukkan bahwa arsitektur yang terlalu besar tidak selalu menguntungkan.

---

## Grafik Output

Program menghasilkan grafik konvergensi otomatis:

| File | Keterangan |
|:-----|:-----------|
| `grafik_and_s1.png` | AND – Skenario 1 (bobot ditentukan) |
| `grafik_and_s2.png` | AND – Skenario 2 (random ±0.5) |
| `grafik_and_s3.png` | AND – Skenario 3 (random ±1.0) |
| `grafik_and_s4.png` | AND – Skenario 4 (multi LR) |
| `grafik_or_s1.png`  | OR – Skenario 1 (bobot ditentukan) |
| `grafik_or_s2.png`  | OR – Skenario 2 (random ±0.5) |
| `grafik_or_s3.png`  | OR – Skenario 3 (random ±1.0) |
| `grafik_or_s4.png`  | OR – Skenario 4 (multi LR) |
| `grafik_xor_s1.png` | XOR – Skenario 1 (bobot ditentukan) |
| `grafik_xor_s2.png` | XOR – Skenario 2 (random ±0.5) |
| `grafik_xor_s3.png` | XOR – Skenario 3 (random ±1.0) |
| `grafik_xor_s4.png` | XOR – Skenario 4 (multi LR) |
| `grafik_xor_s5.png` | XOR – Skenario 5 (hidden node 2 vs 3 vs 5) |

---

*Referensi: Materi P9 Jaringan Syaraf Tiruan – Yisti Vita Via, S.ST., M.Kom. – Informatika UPN Veteran Jawa Timur*
