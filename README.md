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
**UNIVERSITAS PEMBANGUNAN NASIONAL “VETERAN” JAWA TIMUR**  
**2026**

---

## BAB I: PENDAHULUAN

### 1.1 Latar Belakang
Jaringan Syaraf Tiruan (JST) adalah model komputasi yang terinspirasi dari cara kerja otak manusia, terdiri dari neuron-neuron buatan yang saling terhubung, dan mampu belajar dari data untuk mengenali pola serta membuat prediksi. Konsep ini pertama kali diperkenalkan oleh Warren McCulloch dan Walter Pitts pada tahun 1943.

Dalam tugas ini, kami mengimplementasikan JST sederhana untuk menyelesaikan permasalahan logika dasar menggunakan dua jenis arsitektur:
1. Single Layer Perceptron untuk kasus AND dan OR (linear separable)
2. Multilayer Perceptron (MLP) untuk kasus XOR (non-linear separable)

### 1.2 Tujuan
1. Memahami dan mengimplementasikan konsep dasar JST
2. Membangun model Single Layer Perceptron untuk kasus AND dan OR
3. Membangun model MLP untuk kasus XOR
4. Menganalisis perbedaan hasil pada AND, OR, dan XOR
5. Menjelaskan mengapa XOR tidak dapat diselesaikan single layer perceptron

### 1.3 Dataset
Dataset yang digunakan adalah tabel kebenaran logika AND, OR, dan XOR dengan 2 input (x1, x2) dan 1 output (target).

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
Dalam implementasi ini digunakan Step Function:
- `net >= 0` → Output f(net) = 1
- `net < 0` → Output f(net) = 0

**d. Lapisan (Layer)**
1. **Input Layer:** Menerima data masukan (x1, x2)
2. **Hidden Layer:** Lapisan tersembunyi (khusus MLP untuk XOR)
3. **Output Layer:** Menghasilkan prediksi/keputusan

### 2.3 Rumus Perceptron
Rumus perhitungan net input:
`net = w1 × x1 + w2 × x2 + b`

Rumus update bobot (Perceptron Rule):
- `w_baru = w_lama + α × (t - y) × xi`
- `b_baru = b_lama + α × (t - y)`

*(Keterangan: α = learning rate, t = target, y = output JST, xi = input ke-i)*

### 2.4 Tahapan Training JST
1. Inisialisasi bobot dan bias (nilai awal)
2. Forward pass: hitung net dan output y = f(net) untuk setiap data
3. Bandingkan output y dengan target t, hitung error = t - y
4. Jika error ≠ 0, update bobot dan bias menggunakan Perceptron Rule
5. Ulangi langkah 2-4 (epoch) sampai semua output sesuai target

---

## BAB III: DESAIN ARSITEKTUR MODEL

### 3.1 Single Layer Perceptron (AND & OR)
Arsitektur: 2 neuron input → 1 neuron output (tanpa hidden layer)

| Parameter | AND - Nilai | OR - Nilai | Keterangan |
| :--- | :--- | :--- | :--- |
| w1 (bobot x1) | 0.2 | 0.0 | Bobot awal input x1 |
| w2 (bobot x2) | 0.2 | 0.0 | Bobot awal input x2 |
| b (bias) | -0.1 | -0.5 | Bias awal |
| alpha (lr) | 0.1 | 0.5 | Learning rate |
| Aktivasi | Step | Step | f(net)=1 jika net>=0 |

### 3.2 Multilayer Perceptron (XOR)
Arsitektur MLP 2-2-1: 2 neuron input → 2 hidden neuron → 1 neuron output

| Parameter | w11 / w21 | w12 / w22 | v1 / v2 | b1 / b2 / b3 |
| :--- | :--- | :--- | :--- | :--- |
| h1 (OR): Input→H1 | 1.0 / 1.0 | - | - | b1 = -0.5 |
| h2 (NAND): Input→H2 | - | -1.0 / -1.0 | - | b2 = 1.5 |
| Output: H→Output | - | - | 1.0 / 1.0 | b3 = -1.5 |

**Strategi MLP untuk XOR:** h1 bertindak sebagai OR gate, h2 sebagai NAND gate, dan output layer menggabungkannya sebagai AND gate. Kombinasi OR dan NAND menghasilkan XOR.

---

## BAB IV: IMPLEMENTASI & HASIL

### 4.1 Kasus AND - Single Layer Perceptron
**Hasil Training AND** (Konvergen pada epoch 2)
Bobot Akhir AND: `w1 = 0.20, w2 = 0.20, b = -0.20`

### 4.2 Kasus OR - Single Layer Perceptron
**Hasil Training OR** (Konvergen pada epoch 4)
Bobot Akhir OR: `w1 = 0.50, w2 = 0.50, b = -0.50`

### 4.3 Kasus XOR - Multilayer Perceptron
**Forward Pass MLP XOR - Detail Perhitungan**

| No | x1 | x2 | t | net_h1 | h1 | net_h2 | h2 | net_y | y | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0 | 0 | 0 | -0.5 | 0 | 1.5 | 1 | -0.5 | 0 | ✓ Benar |
| 2 | 0 | 1 | 1 | 0.5 | 1 | 0.5 | 1 | 0.5 | 1 | ✓ Benar |
| 3 | 1 | 0 | 1 | 0.5 | 1 | 0.5 | 1 | 0.5 | 1 | ✓ Benar |
| 4 | 1 | 1 | 0 | 1.5 | 1 | -0.5 | 0 | -0.5 | 0 | ✓ Benar |

Hasil: Semua 4 data XOR diprediksi dengan benar oleh MLP! Akurasi 100%.

---

## BAB V: ANALISIS HASIL

### 5.1 Perbandingan Hasil AND, OR, dan XOR

| Kasus | Model | Epoch | Akurasi | Keterangan |
| :--- | :--- | :--- | :--- | :--- |
| AND | Single Layer | 2 epoch | 100% | Linear separable |
| OR | Single Layer | 4 epoch | 100% | Linear separable |
| XOR (single) | Single Layer | Tidak konvergen | <100% | Non-linear separable |
| XOR (MLP) | 2 Hidden Neuron | Langsung | 100% | Butuh hidden layer |

### 5.2 Perbedaan AND, OR, dan XOR
Perbedaan utama antara ketiga kasus terletak pada sifat linear separability-nya:
1. **AND:** Hanya bernilai 1 jika KEDUA input bernilai 1. Titik (1,1) terpisah dari yang lain dengan garis lurus → *linear separable* → Single Layer cukup.
2. **OR:** Bernilai 1 jika MINIMAL SATU input bernilai 1. Hanya titik (0,0) yang bernilai 0, sisanya 1 → *linear separable* → Single Layer cukup.
3. **XOR:** Bernilai 1 jika input BERBEDA. Titik (0,1) dan (1,0) bernilai 1, sedangkan (0,0) dan (1,1) bernilai 0. Pola ini membentuk 'checkerboard' yang TIDAK bisa dipisahkan dengan satu garis lurus.

### 5.3 Mengapa XOR Tidak Bisa Diselesaikan Single Layer
Berdasarkan materi P9 JST, terdapat tiga alasan utama:
1. **XOR bersifat NON-LINEARLY SEPARABLE.** Tidak ada satupun garis lurus (hyperplane) yang dapat memisahkan titik XOR=1 dari XOR=0 dalam ruang 2 dimensi.
2. **Single Layer Perceptron hanya dapat membentuk batas keputusan LINIER (garis lurus).** Ini terbukti dari percobaan kita: pelatihan tidak konvergen meskipun dijalankan banyak epoch.
3. Keterbatasan ini pertama kali dibuktikan oleh Marvin Minsky dan Seymour Papert dalam buku 'Perceptrons' (1969), yang menyebabkan *AI Winter* pada masa itu.

Solusi: MLP dengan hidden layer dapat membentuk batas keputusan NON-LINEAR karena hidden layer 'mentransformasi' ruang input ke representasi baru yang linear separable.

---

## BAB VI: KESIMPULAN

1. Single Layer Perceptron berhasil menyelesaikan kasus AND (konvergen epoch ke-2) dan OR (konvergen epoch ke-4) karena keduanya bersifat *linear separable*.
2. Single Layer Perceptron TIDAK MAMPU menyelesaikan XOR karena XOR bersifat *non-linearly separable* (tidak ada garis lurus yang bisa memisahkan kelasnya).
3. Multilayer Perceptron (MLP) dengan 2 hidden neuron berhasil menyelesaikan XOR dengan akurasi 100%. Hidden layer h1 bertindak sebagai OR gate dan h2 sebagai NAND gate, yang dikombinasikan menghasilkan XOR.
4. Semakin banyak hidden layer → semakin kompleks masalah yang bisa diselesaikan (ini merupakan dasar *Deep Learning*).
5. Pemilihan arsitektur JST yang tepat (jumlah layer, neuron, fungsi aktivasi, learning rate) sangat menentukan keberhasilan model.

---
*Referensi: Materi P9 Jaringan Syaraf Tiruan - Yisti Vita Via, S.ST., M.Kom. - Informatika UPN Veteran Jawa Timur*
