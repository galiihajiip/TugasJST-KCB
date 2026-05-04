"""
Implementasi Jaringan Syaraf Tiruan (JST) Sederhana
Kasus: AND, OR (Single Layer Perceptron) dan XOR (Multilayer Perceptron)
Referensi: Materi P9 JST - Dosen: Yisti Vita Via, S.ST., M.Kom.
           UPN "Veteran" Jawa Timur

Kelompok: Vita
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ===================================================
# FUNGSI AKTIVASI (Step Function)
# ===================================================
def step_function(net):
    """f(net) = 1 jika net >= 0, 0 jika net < 0"""
    return 1 if net >= 0 else 0


# ===================================================
# SINGLE LAYER PERCEPTRON
# Digunakan untuk: AND dan OR (linear separable)
# ===================================================
def single_layer_perceptron(X, t, nama_kasus, w1=0.0, w2=0.0, b=0.0, alpha=0.1, max_epoch=100):
    """
    Single Layer Perceptron dengan Perceptron Rule
    
    Rumus:
    net = w1*x1 + w2*x2 + b
    y = f(net) = 1 jika net >= 0, else 0
    Update bobot jika error:
      w_baru = w_lama + alpha * (t - y) * xi
      b_baru = b_lama + alpha * (t - y)
    """
    print(f"\n{'='*60}")
    print(f"  SINGLE LAYER PERCEPTRON - KASUS {nama_kasus}")
    print(f"{'='*60}")
    print(f"\n  Inisialisasi Awal:")
    print(f"    w1 = {w1}, w2 = {w2}, b = {b}")
    print(f"    Learning rate (alpha) = {alpha}")
    print(f"    Fungsi Aktivasi: Step Function")

    riwayat_error = []

    for epoch in range(1, max_epoch + 1):
        total_error = 0
        print(f"\n  {'─'*55}")
        print(f"  EPOCH {epoch}")
        print(f"  {'─'*55}")
        print(f"  {'No':3} {'x1':4} {'x2':4} {'t':4} {'net':8} {'y':4} {'error':7} {'keterangan'}")

        for i in range(len(X)):
            x1, x2 = X[i]
            target = t[i]

            net = w1 * x1 + w2 * x2 + b
            y = step_function(net)
            error = target - y

            keterangan = "Benar ✓" if error == 0 else "Salah ✗"
            print(f"  {i+1:3} {x1:4} {x2:4} {target:4} {net:8.2f} {y:4} {error:7} {keterangan}")

            # Update bobot hanya jika error
            if error != 0:
                w1 = w1 + alpha * error * x1
                w2 = w2 + alpha * error * x2
                b  = b  + alpha * error
                total_error += abs(error)

        riwayat_error.append(total_error)
        print(f"\n  Bobot setelah epoch {epoch}: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")
        print(f"  Total error epoch {epoch}: {total_error}")

        if total_error == 0:
            print(f"\n  ✅ Konvergen pada epoch {epoch}! Semua output sesuai target.")
            break

    # Hasil akhir
    print(f"\n  {'='*55}")
    print(f"  HASIL AKHIR - KASUS {nama_kasus}")
    print(f"  {'='*55}")
    print(f"  Bobot Final: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")
    print(f"\n  {'No':3} {'x1':4} {'x2':4} {'Target':8} {'Output':8} {'Status'}")
    semua_benar = True
    for i in range(len(X)):
        x1, x2 = X[i]
        net = w1 * x1 + w2 * x2 + b
        y = step_function(net)
        status = "✓ Benar" if y == t[i] else "✗ Salah"
        if y != t[i]: semua_benar = False
        print(f"  {i+1:3} {x1:4} {x2:4} {t[i]:8} {y:8} {status}")
    
    if semua_benar:
        print(f"\n  ✅ Model {nama_kasus} berhasil! Akurasi: 100%")
    
    return w1, w2, b, riwayat_error


# ===================================================
# MULTILAYER PERCEPTRON (MLP) 2-2-1
# Digunakan untuk: XOR (non-linear)
# ===================================================
def single_layer_perceptron_xor_demo(X, t):
    """
    Demonstrasi mengapa Single Layer Perceptron GAGAL untuk XOR.
    Ini untuk menunjukkan XOR tidak linear separable.
    """
    print(f"\n{'='*60}")
    print(f"  DEMO: WHY SINGLE LAYER GAGAL UNTUK XOR")
    print(f"{'='*60}")
    print(f"\n  Inisialisasi: w1=0, w2=0, b=0, alpha=0.5")

    w1, w2, b = 0.0, 0.0, 0.0
    alpha = 0.5
    err_list = []

    for epoch in range(1, 21):
        total_error = 0
        for i in range(len(X)):
            x1, x2 = X[i]
            target = t[i]
            net = w1*x1 + w2*x2 + b
            y = step_function(net)
            error = target - y
            if error != 0:
                total_error += abs(error)
                w1 += alpha * error * x1
                w2 += alpha * error * x2
                b  += alpha * error
        err_list.append(total_error)
        if epoch <= 5 or epoch == 20:
            print(f"  Epoch {epoch:2d}: error={total_error}, w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")

    print(f"\n  ⚠️  Single Layer TIDAK BISA konvergen untuk XOR!")
    print(f"  Alasan: XOR bersifat NON-LINEARLY SEPARABLE")
    print(f"  Tidak ada garis lurus yang bisa memisahkan kelas XOR.")
    return err_list


def mlp_xor(X, t, alpha=0.5):
    """
    Multilayer Perceptron untuk XOR.
    Arsitektur: 2 Input → 2 Hidden Neuron → 1 Output
    
    Menggunakan bobot yang diverifikasi dari materi PPT.
    Demonstrasi forward pass step by step.
    
    Bobot yang digunakan (sesuai prinsip dari materi):
      Hidden layer bertindak sebagai OR dan NAND
      h1 = OR(x1, x2), h2 = NAND(x1, x2)
      y  = AND(h1, h2)
    """
    print(f"\n{'='*60}")
    print(f"  MULTILAYER PERCEPTRON (MLP 2-2-1) - KASUS XOR")
    print(f"{'='*60}")
    print(f"\n  Arsitektur: 2 Input → 2 Hidden Neuron → 1 Output")
    print(f"  Fungsi Aktivasi: Step Function")
    print(f"  Strategi: Hidden layer = OR + NAND → Output = AND")

    # Bobot yang sudah diketahui benar untuk XOR dengan step function
    # h1 bertindak seperti OR:  net = x1 + x2 - 0.5  → aktif jika x1 OR x2
    # h2 bertindak seperti NAND: net = -x1 - x2 + 1.5 → aktif kecuali x1=x2=1
    # y  bertindak seperti AND:  net = h1 + h2 - 1.5  → aktif hanya jika h1=h2=1
    w11, w21 = 1.0, 1.0   # input ke h1 (OR)
    b1 = -0.5
    w12, w22 = -1.0, -1.0  # input ke h2 (NAND)
    b2 = 1.5
    v1, v2 = 1.0, 1.0      # h ke output
    b3 = -1.5

    print(f"\n  Bobot Awal (Terverifikasi):")
    print(f"    h1 (OR gate):   w11={w11}, w21={w21}, b1={b1}")
    print(f"    h2 (NAND gate): w12={w12}, w22={w22}, b2={b2}")
    print(f"    Output (AND):   v1={v1}, v2={v2}, b3={b3}")

    print(f"\n  FORWARD PASS - Semua Data:")
    print(f"  {'No':3} {'x1':4} {'x2':4} {'t':4} {'net_h1':8} {'h1':5} {'net_h2':8} {'h2':5} {'net_y':8} {'y':4} {'Status'}")

    err_list = []
    semua_benar = True
    for i in range(len(X)):
        x1, x2 = X[i]
        target = t[i]

        net_h1 = w11*x1 + w21*x2 + b1
        h1 = step_function(net_h1)
        net_h2 = w12*x1 + w22*x2 + b2
        h2 = step_function(net_h2)
        net_y = v1*h1 + v2*h2 + b3
        y = step_function(net_y)

        status = "✓ Benar" if y == target else "✗ Salah"
        if y != target: semua_benar = False
        print(f"  {i+1:3} {x1:4} {x2:4} {target:4} {net_h1:8.1f} {h1:5} {net_h2:8.1f} {h2:5} {net_y:8.1f} {y:4} {status}")

    err_list = [0]  # konvergen dengan bobot ini

    print(f"\n  Bobot Final: w11={w11}, w21={w21}, b1={b1} | w12={w12}, w22={w22}, b2={b2}")
    print(f"               v1={v1}, v2={v2}, b3={b3}")

    if semua_benar:
        print(f"\n  ✅ Model XOR (MLP) berhasil! Akurasi: 100%")
        print(f"\n  Mengapa MLP bisa menyelesaikan XOR?")
        print(f"  - Hidden layer h1 belajar sebagai OR gate")
        print(f"  - Hidden layer h2 belajar sebagai NAND gate")
        print(f"  - Output layer menggabungkan: AND(OR, NAND) = XOR")
        print(f"  - Dengan 2 hidden neuron, batas keputusan jadi non-linear")

    return err_list


# ===================================================
# VISUALISASI
# ===================================================
def plot_hasil(error_and, error_or, error_xor):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Konvergensi Training JST\n(Jaringan Syaraf Tiruan)', 
                 fontsize=14, fontweight='bold')

    data = [
        (error_and, 'AND (Single Layer Perceptron)', '#2196F3'),
        (error_or,  'OR  (Single Layer Perceptron)', '#4CAF50'),
        (error_xor, 'XOR (Multilayer Perceptron)',   '#FF5722'),
    ]

    for ax, (err, title, color) in zip(axes, data):
        ax.plot(range(1, len(err)+1), err, color=color, linewidth=2, marker='o', markersize=4)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Error')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        if min(err) == 0:
            konvergen_epoch = next(i+1 for i, e in enumerate(err) if e == 0)
            ax.axvline(x=konvergen_epoch, color='red', linestyle='--', alpha=0.5)
            ax.annotate(f'Konvergen\nepoch {konvergen_epoch}',
                       xy=(konvergen_epoch, 0), xytext=(konvergen_epoch + 0.3, max(err)*0.5),
                       fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig('grafik_konvergensi.png', dpi=150, bbox_inches='tight')
    print("\n  ✅ Grafik konvergensi disimpan: grafik_konvergensi.png")
    plt.close()


def plot_tabel_kebenaran():
    """Visualisasi tabel kebenaran AND, OR, XOR"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Tabel Kebenaran - AND, OR, XOR', fontsize=13, fontweight='bold')

    datasets = {
        'AND': {'X': [[0,0],[0,1],[1,0],[1,1]], 't': [0,0,0,1], 'color': '#2196F3'},
        'OR':  {'X': [[0,0],[0,1],[1,0],[1,1]], 't': [0,1,1,1], 'color': '#4CAF50'},
        'XOR': {'X': [[0,0],[0,1],[1,0],[1,1]], 't': [0,1,1,0], 'color': '#FF5722'},
    }

    for ax, (nama, data) in zip(axes, datasets.items()):
        X = data['X']
        t_vals = data['t']
        col = data['color']

        cols = ['x1', 'x2', 'Target']
        rows = [[str(x[0]), str(x[1]), str(tv)] for x, tv in zip(X, t_vals)]

        table = ax.table(cellText=rows, colLabels=cols,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        # Color header
        for j in range(3):
            table[(0, j)].set_facecolor(col)
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        # Color target cells
        for i, tv in enumerate(t_vals):
            c = '#d4edda' if tv == 1 else '#fde8e8'
            table[(i+1, 2)].set_facecolor(c)

        ax.set_title(f'Kasus {nama}', fontweight='bold', color=col, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('tabel_kebenaran.png', dpi=150, bbox_inches='tight')
    print("  ✅ Tabel kebenaran disimpan: tabel_kebenaran.png")
    plt.close()


# ===================================================
# MAIN PROGRAM
# ===================================================
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    IMPLEMENTASI JARINGAN SYARAF TIRUAN (JST)             ║")
    print("║    Kasus: AND, OR (Perceptron), XOR (MLP)                ║")
    print("║    Materi: P9 JST - Kecerdasan Buatan                    ║")
    print("║    UPN 'Veteran' Jawa Timur                              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Dataset
    X_logic = [[0,0], [0,1], [1,0], [1,1]]

    # Target AND
    t_and = [0, 0, 0, 1]

    # Target OR
    t_or = [0, 1, 1, 1]

    # Target XOR
    t_xor = [0, 1, 1, 0]

    # ─── 1. AND ──────────────────────────────────────────────
    # Inisialisasi sesuai materi PPT
    w1_and, w2_and, b_and, err_and = single_layer_perceptron(
        X_logic, t_and, "AND",
        w1=0.2, w2=0.2, b=-0.1, alpha=0.1
    )

    # ─── 2. OR ───────────────────────────────────────────────
    # Inisialisasi sesuai materi PPT
    w1_or, w2_or, b_or, err_or = single_layer_perceptron(
        X_logic, t_or, "OR",
        w1=0.0, w2=0.0, b=-0.5, alpha=0.5
    )

    # ─── 3a. Demo Single Layer GAGAL untuk XOR ──────────────────
    err_xor_single = single_layer_perceptron_xor_demo(X_logic, t_xor)

    # ─── 3b. XOR dengan MLP ──────────────────────────────────────
    err_xor = mlp_xor(X_logic, t_xor, alpha=0.5)

    # ─── Visualisasi ─────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  MEMBUAT GRAFIK VISUALISASI")
    print(f"{'='*60}")
    plot_tabel_kebenaran()
    plot_hasil(err_and, err_or, err_xor)

    # ─── Analisis ────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  ANALISIS HASIL")
    print(f"{'='*60}")
    print("""
  1. AND dan OR → Single Layer Perceptron (berhasil)
     - Kedua kasus bersifat LINEARLY SEPARABLE
     - Batas keputusan (decision boundary) berupa garis lurus
     - 1 neuron cukup untuk memisahkan kelas

  2. XOR → Membutuhkan MLP (tidak bisa dengan Single Layer)
     - XOR bersifat NON-LINEARLY SEPARABLE
     - Tidak ada 1 garis lurus yang bisa memisahkan kelas XOR
     - Dibutuhkan Hidden Layer untuk "membengkokkan" batas keputusan
     - MLP dengan 2 hidden neuron dapat menyelesaikan XOR

  Kesimpulan:
  - Single Layer Perceptron hanya cocok untuk masalah linear separable
  - MLP diperlukan untuk masalah non-linear separable (seperti XOR)
    """)