# -*- coding: utf-8 -*-
"""
Implementasi Jaringan Syaraf Tiruan (JST)
Kasus: AND, OR (Single Layer Perceptron) dan XOR (Multilayer Perceptron)
Skenario percobaan sesuai arahan dosen.

Skenario AND & OR:
  1. Bobot awal ditentukan, LR = 0.1
  2. Bobot awal random -0.5 s.d 0.5, LR = 0.1
  3. Bobot awal random -1.0 s.d 1.0, LR = 0.1
  4. Bobot awal ditentukan, LR = 0.01 / 0.05 / 0.1 / 0.5

Skenario XOR:
  1. Bobot awal ditentukan, LR = 0.1
  2. Bobot awal random -0.5 s.d 0.5, LR = 0.1
  3. Bobot awal random -1.0 s.d 1.0, LR = 0.1
  4. Bobot awal ditentukan, LR = 0.01 / 0.05 / 0.1 / 0.5
  5. Bobot awal random -0.5 s.d 0.5, LR = 0.1, Hidden Node 3 dan 5
"""

import numpy as np
import matplotlib.pyplot as plt

# ===================================================
# FUNGSI AKTIVASI
# ===================================================
def step_function(net):
    """Step: 1 jika net >= 0, else 0"""
    return 1 if net >= 0 else 0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# ===================================================
# SINGLE LAYER PERCEPTRON (AND / OR)
# ===================================================
def run_slp(X, t, w1, w2, b, alpha, max_epoch=1000, verbose=True, nama=""):
    """
    Jalankan Single Layer Perceptron.
    Kembalikan (w1_final, w2_final, b_final, riwayat_error, konvergen)
    """
    if verbose:
        print(f"\n  Inisialisasi: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}, alpha={alpha}")
        print(f"  {'No':3} {'Ep':4} {'x1':4} {'x2':4} {'t':4} {'net':8} {'y':4} {'err':5} {'w1':8} {'w2':8} {'b':8}")

    riwayat_error = []
    konvergen = False

    for epoch in range(1, max_epoch + 1):
        total_error = 0
        for i in range(len(X)):
            x1, x2 = float(X[i][0]), float(X[i][1])
            target = t[i]
            net = w1 * x1 + w2 * x2 + b
            y = step_function(net)
            error = target - y
            if verbose:
                print(f"  {i+1:3} {epoch:4} {int(x1):4} {int(x2):4} {target:4} "
                      f"{net:8.4f} {y:4} {error:5} {w1:8.4f} {w2:8.4f} {b:8.4f}")
            if error != 0:
                w1 = w1 + alpha * error * x1
                w2 = w2 + alpha * error * x2
                b  = b  + alpha * error
                total_error += abs(error)

        riwayat_error.append(total_error)
        if verbose:
            print(f"  >> Akhir Epoch {epoch}: total_error={total_error}, "
                  f"w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}")
        if total_error == 0:
            konvergen = True
            break

    return w1, w2, b, riwayat_error, konvergen


def print_slp_result(X, t, w1, w2, b, nama, riwayat_error, konvergen, max_epoch):
    """Cetak hasil akhir SLP."""
    print(f"\n  {'='*55}")
    print(f"  HASIL AKHIR - {nama}")
    print(f"  Bobot Final: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}")
    if konvergen:
        print(f"  Status: KONVERGEN pada Epoch {len(riwayat_error)}")
    else:
        print(f"  Status: BELUM KONVERGEN (>{max_epoch} epoch)")
    print(f"\n  {'No':3} {'x1':4} {'x2':4} {'Target':8} {'Output':8} {'Status'}")
    for i in range(len(X)):
        x1, x2 = float(X[i][0]), float(X[i][1])
        net = w1*x1 + w2*x2 + b
        y = step_function(net)
        status = "Benar" if y == t[i] else "SALAH"
        print(f"  {i+1:3} {int(x1):4} {int(x2):4} {t[i]:8} {y:8} {status}")
    akurasi = sum(
        step_function(w1*float(X[i][0]) + w2*float(X[i][1]) + b) == t[i]
        for i in range(len(X))
    ) / len(X) * 100
    print(f"  Akurasi: {akurasi:.0f}%")


# ===================================================
# MLP BACKPROPAGATION (XOR)
# ===================================================
def run_mlp(X, t, hidden_nodes, w_range, alpha, max_epoch=10000, seed=None, verbose=False):
    """
    MLP 2 → hidden_nodes → 1 dengan backpropagation.
    Aktivasi: sigmoid. Konvergen jika MSE < 0.01.
    """
    rng = np.random.RandomState(seed)
    W1 = rng.uniform(-w_range, w_range, (2, hidden_nodes))
    b1 = rng.uniform(-w_range, w_range, (hidden_nodes,))
    W2 = rng.uniform(-w_range, w_range, (hidden_nodes, 1))
    b2 = rng.uniform(-w_range, w_range, (1,))

    X_arr = np.array(X, dtype=float)
    t_arr = np.array(t, dtype=float).reshape(-1, 1)
    riwayat_mse = []
    konvergen = False

    for epoch in range(1, max_epoch + 1):
        net_h = X_arr @ W1 + b1
        H     = sigmoid(net_h)
        net_o = H @ W2 + b2
        Y     = sigmoid(net_o)
        diff  = t_arr - Y
        mse   = float(np.mean(diff**2))
        riwayat_mse.append(mse)

        delta_o = diff * sigmoid_deriv(net_o)
        dW2 = H.T @ delta_o;  db2 = np.sum(delta_o, axis=0)
        delta_h = (delta_o @ W2.T) * sigmoid_deriv(net_h)
        dW1 = X_arr.T @ delta_h; db1_d = np.sum(delta_h, axis=0)

        W1 += alpha * dW1;  b1 += alpha * db1_d
        W2 += alpha * dW2;  b2 += alpha * db2

        if verbose and (epoch <= 5 or epoch % 1000 == 0):
            print(f"    Epoch {epoch:5d}: MSE={mse:.6f}")

        if mse < 0.01:
            konvergen = True
            break

    # Prediksi final
    Y_pred = sigmoid(sigmoid(X_arr @ W1 + b1) @ W2 + b2).flatten()
    Y_bin  = (Y_pred >= 0.5).astype(int)
    akurasi = sum(Y_bin == np.array(t)) / len(t) * 100
    return riwayat_mse, Y_bin, Y_pred, konvergen, len(riwayat_mse), akurasi


def print_mlp_result(X, t, Y_bin, Y_pred, konvergen, n_epoch, akurasi, arsitektur, max_epoch):
    """Cetak hasil akhir MLP."""
    print(f"\n  Arsitektur : {arsitektur}")
    if konvergen:
        print(f"  Status     : KONVERGEN pada Epoch {n_epoch}")
    else:
        print(f"  Status     : BELUM KONVERGEN (>{max_epoch} epoch)")
    print(f"  Akurasi    : {akurasi:.0f}%")
    print(f"\n  {'No':3} {'x1':4} {'x2':4} {'Target':8} {'Sigmoid':10} {'Biner':7} {'Status'}")
    for i in range(len(X)):
        x1, x2 = int(X[i][0]), int(X[i][1])
        status = "Benar" if Y_bin[i] == t[i] else "SALAH"
        print(f"  {i+1:3} {x1:4} {x2:4} {t[i]:8} {Y_pred[i]:10.4f} {Y_bin[i]:7} {status}")


# ===================================================
# VISUALISASI GRAFIK KONVERGENSI
# ===================================================
def plot_konvergensi_multi(data_list, judul_besar, filename):
    """
    data_list: list of (error_list, label, color)
    """
    n = len(data_list)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(judul_besar, fontsize=12, fontweight='bold')

    for ax, (err, label, color) in zip(axes, data_list):
        ax.plot(range(1, len(err)+1), err, color=color, linewidth=1.5, marker='o', markersize=3)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error / MSE")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        # Tandai konvergensi
        last = err[-1]
        if last == 0 or last < 0.01:
            ax.axvline(x=len(err), color='red', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  >> Grafik disimpan: {filename}")


# ===================================================
# SKENARIO SLP (AND / OR)
# ===================================================
def skenario_slp(kasus, X, t):
    WARNA = ['#2196F3','#4CAF50','#FF9800','#E91E63']
    LR_LIST = [0.01, 0.05, 0.1, 0.5]
    MAX_EPOCH = 1000

    # Bobot ditentukan yang sama untuk semua skenario bobot-tetap
    if kasus == "AND":
        W1_TETAP, W2_TETAP, B_TETAP = 0.2, 0.2, -0.1
    else:  # OR
        W1_TETAP, W2_TETAP, B_TETAP = 0.0, 0.0, -0.5

    print(f"\n{'#'*62}")
    print(f"#  KASUS {kasus} - SINGLE LAYER PERCEPTRON")
    print(f"{'#'*62}")

    # ── Skenario 1: Bobot ditentukan, LR = 0.1 ──────────────────
    header("Skenario 1", f"Bobot ditentukan, LR = 0.1 (bandingkan dg Excel)")
    print(f"  Bobot awal: w1={W1_TETAP}, w2={W2_TETAP}, b={B_TETAP}")
    w1f, w2f, bf, err, konv = run_slp(
        X, t, W1_TETAP, W2_TETAP, B_TETAP, 0.1, MAX_EPOCH, verbose=True
    )
    print_slp_result(X, t, w1f, w2f, bf, f"{kasus} S1", err, konv, MAX_EPOCH)
    plot_konvergensi_multi(
        [(err, f"{kasus} – Skenario 1 (LR=0.1)", '#2196F3')],
        f"Konvergensi {kasus} – Skenario 1",
        f"grafik_{kasus.lower()}_s1.png"
    )

    # ── Skenario 2: Random -0.5 s.d 0.5, LR = 0.1 ───────────────
    header("Skenario 2", "Bobot random -0.5 s.d 0.5, LR = 0.1")
    data_plot = []
    for run_i in range(3):
        rng = np.random.RandomState(run_i * 7 + 13)
        w1i = rng.uniform(-0.5, 0.5)
        w2i = rng.uniform(-0.5, 0.5)
        bi  = rng.uniform(-0.5, 0.5)
        print(f"\n  [Run {run_i+1}] w1={w1i:.4f}, w2={w2i:.4f}, b={bi:.4f}")
        w1f, w2f, bf, err, konv = run_slp(
            X, t, w1i, w2i, bi, 0.1, MAX_EPOCH, verbose=True
        )
        print_slp_result(X, t, w1f, w2f, bf, f"{kasus} S2-Run{run_i+1}", err, konv, MAX_EPOCH)
        data_plot.append((err, f"Run {run_i+1}", WARNA[run_i]))
    plot_konvergensi_multi(
        data_plot,
        f"Konvergensi {kasus} – Skenario 2 (Random ±0.5, LR=0.1)",
        f"grafik_{kasus.lower()}_s2.png"
    )

    # ── Skenario 3: Random -1.0 s.d 1.0, LR = 0.1 ───────────────
    header("Skenario 3", "Bobot random -1.0 s.d 1.0, LR = 0.1")
    data_plot = []
    for run_i in range(3):
        rng = np.random.RandomState(run_i * 11 + 5)
        w1i = rng.uniform(-1.0, 1.0)
        w2i = rng.uniform(-1.0, 1.0)
        bi  = rng.uniform(-1.0, 1.0)
        print(f"\n  [Run {run_i+1}] w1={w1i:.4f}, w2={w2i:.4f}, b={bi:.4f}")
        w1f, w2f, bf, err, konv = run_slp(
            X, t, w1i, w2i, bi, 0.1, MAX_EPOCH, verbose=True
        )
        print_slp_result(X, t, w1f, w2f, bf, f"{kasus} S3-Run{run_i+1}", err, konv, MAX_EPOCH)
        data_plot.append((err, f"Run {run_i+1}", WARNA[run_i]))
    plot_konvergensi_multi(
        data_plot,
        f"Konvergensi {kasus} – Skenario 3 (Random ±1.0, LR=0.1)",
        f"grafik_{kasus.lower()}_s3.png"
    )

    # ── Skenario 4: Bobot ditentukan, multi LR ───────────────────
    header("Skenario 4", "Bobot ditentukan, LR = 0.01 / 0.05 / 0.1 / 0.5")
    print(f"  Bobot awal: w1={W1_TETAP}, w2={W2_TETAP}, b={B_TETAP}")
    data_plot = []
    ringkasan = []
    for idx, lr in enumerate(LR_LIST):
        print(f"\n  --- LR = {lr} ---")
        w1f, w2f, bf, err, konv = run_slp(
            X, t, W1_TETAP, W2_TETAP, B_TETAP, lr, MAX_EPOCH, verbose=False
        )
        ep_konv = len(err) if konv else None
        print(f"  {'Konvergen' if konv else 'Belum konvergen'} | "
              f"Epoch: {ep_konv if konv else '>'+str(MAX_EPOCH)} | "
              f"w1={w1f:.4f}, w2={w2f:.4f}, b={bf:.4f}")
        ringkasan.append((lr, ep_konv, w1f, w2f, bf, konv))
        data_plot.append((err, f"LR={lr}", WARNA[idx]))

    print(f"\n  {'LR':8} {'Epoch':10} {'W1 Final':10} {'W2 Final':10} {'b Final':10} {'Konvergen'}")
    for lr, ep, w1f, w2f, bf, konv in ringkasan:
        ep_str = str(ep) if konv else f">{MAX_EPOCH}"
        print(f"  {lr:<8} {ep_str:<10} {w1f:<10.4f} {w2f:<10.4f} {bf:<10.4f} {'Ya' if konv else 'Tidak'}")

    plot_konvergensi_multi(
        data_plot,
        f"Konvergensi {kasus} – Skenario 4 (Multi LR)",
        f"grafik_{kasus.lower()}_s4.png"
    )


# ===================================================
# SKENARIO MLP XOR
# ===================================================
def skenario_xor(X, t):
    WARNA = ['#2196F3','#4CAF50','#FF9800','#E91E63']
    LR_LIST  = [0.01, 0.05, 0.1, 0.5]
    MAX_EPOCH = 10000

    print(f"\n{'#'*62}")
    print(f"#  KASUS XOR - MULTILAYER PERCEPTRON (Backpropagation)")
    print(f"{'#'*62}")
    print(f"  Fungsi aktivasi: Sigmoid | Konvergen jika MSE < 0.01")

    # ── Skenario 1: Bobot ditentukan, LR = 0.1 ───────────────────
    header("Skenario 1 XOR", "Bobot ditentukan, LR = 0.1 (bandingkan dg Excel)")
    # Bobot awal seed tetap 42, range 0.5
    err, Y_bin, Y_pred, konv, n_ep, acc = run_mlp(
        X, t, hidden_nodes=2, w_range=0.5, alpha=0.1,
        max_epoch=MAX_EPOCH, seed=42, verbose=True
    )
    print_mlp_result(X, t, Y_bin, Y_pred, konv, n_ep, acc, "2-2-1", MAX_EPOCH)
    plot_konvergensi_multi(
        [(err, "XOR – Skenario 1 (LR=0.1)", '#FF5722')],
        "Konvergensi XOR – Skenario 1",
        "grafik_xor_s1.png"
    )

    # ── Skenario 2: Random -0.5 s.d 0.5, LR = 0.1 ───────────────
    header("Skenario 2 XOR", "Bobot random -0.5 s.d 0.5, LR = 0.1")
    data_plot = []
    for run_i in range(3):
        seed_i = run_i * 17 + 3
        print(f"\n  [Run {run_i+1}] seed={seed_i}")
        err, Y_bin, Y_pred, konv, n_ep, acc = run_mlp(
            X, t, hidden_nodes=2, w_range=0.5, alpha=0.1,
            max_epoch=MAX_EPOCH, seed=seed_i, verbose=True
        )
        print_mlp_result(X, t, Y_bin, Y_pred, konv, n_ep, acc, "2-2-1", MAX_EPOCH)
        data_plot.append((err, f"Run {run_i+1}", WARNA[run_i]))
    plot_konvergensi_multi(
        data_plot,
        "Konvergensi XOR – Skenario 2 (Random ±0.5, LR=0.1)",
        "grafik_xor_s2.png"
    )

    # ── Skenario 3: Random -1.0 s.d 1.0, LR = 0.1 ───────────────
    header("Skenario 3 XOR", "Bobot random -1.0 s.d 1.0, LR = 0.1")
    data_plot = []
    for run_i in range(3):
        seed_i = run_i * 23 + 7
        print(f"\n  [Run {run_i+1}] seed={seed_i}")
        err, Y_bin, Y_pred, konv, n_ep, acc = run_mlp(
            X, t, hidden_nodes=2, w_range=1.0, alpha=0.1,
            max_epoch=MAX_EPOCH, seed=seed_i, verbose=True
        )
        print_mlp_result(X, t, Y_bin, Y_pred, konv, n_ep, acc, "2-2-1", MAX_EPOCH)
        data_plot.append((err, f"Run {run_i+1}", WARNA[run_i]))
    plot_konvergensi_multi(
        data_plot,
        "Konvergensi XOR – Skenario 3 (Random ±1.0, LR=0.1)",
        "grafik_xor_s3.png"
    )

    # ── Skenario 4: Bobot ditentukan, multi LR ───────────────────
    header("Skenario 4 XOR", "Bobot ditentukan (seed=42), LR = 0.01 / 0.05 / 0.1 / 0.5")
    data_plot = []
    ringkasan = []
    for idx, lr in enumerate(LR_LIST):
        print(f"\n  --- LR = {lr} ---")
        err, Y_bin, Y_pred, konv, n_ep, acc = run_mlp(
            X, t, hidden_nodes=2, w_range=0.5, alpha=lr,
            max_epoch=MAX_EPOCH, seed=42, verbose=True
        )
        ep_str = str(n_ep) if konv else f">{MAX_EPOCH}"
        print(f"  Status: {'Konvergen' if konv else 'Belum konvergen'} | "
              f"Epoch: {ep_str} | Akurasi: {acc:.0f}% | MSE akhir: {err[-1]:.6f}")
        ringkasan.append((lr, ep_str, acc, err[-1], konv))
        data_plot.append((err, f"LR={lr}", WARNA[idx]))

    print(f"\n  {'LR':8} {'Epoch':12} {'Akurasi':10} {'MSE Akhir':12} {'Konvergen'}")
    for lr, ep_str, acc, mse_akhir, konv in ringkasan:
        print(f"  {lr:<8} {ep_str:<12} {acc:<10.0f} {mse_akhir:<12.6f} {'Ya' if konv else 'Tidak'}")

    plot_konvergensi_multi(
        data_plot,
        "Konvergensi XOR – Skenario 4 (Multi LR)",
        "grafik_xor_s4.png"
    )

    # ── Skenario 5: Hidden Node 2, 3, dan 5 ──────────────────────
    header("Skenario 5 XOR", "Random ±0.5, LR = 0.1, Hidden Node = 2, 3, 5")
    hidden_configs = [2, 3, 5]
    arsitektur_labels = ["2-2-1", "2-3-1", "2-5-1"]
    WARNA5 = ['#2196F3', '#4CAF50', '#FF5722']
    data_plot = []
    ringkasan5 = []

    for hn, label in zip(hidden_configs, arsitektur_labels):
        print(f"\n  === Arsitektur {label} (Hidden Nodes = {hn}) ===")
        err, Y_bin, Y_pred, konv, n_ep, acc = run_mlp(
            X, t, hidden_nodes=hn, w_range=0.5, alpha=0.1,
            max_epoch=MAX_EPOCH, seed=42, verbose=True
        )
        print_mlp_result(X, t, Y_bin, Y_pred, konv, n_ep, acc, label, MAX_EPOCH)
        data_plot.append((err, label, WARNA5[hidden_configs.index(hn)]))
        ep_str = str(n_ep) if konv else f">{MAX_EPOCH}"
        ringkasan5.append((label, hn, ep_str, acc, err[-1], konv))

    print(f"\n  {'Arsitektur':12} {'Hidden':8} {'Epoch':12} {'Akurasi':10} {'MSE Akhir':12} {'Konvergen'}")
    for lbl, hn, ep_str, acc, mse_akhir, konv in ringkasan5:
        print(f"  {lbl:<12} {hn:<8} {ep_str:<12} {acc:<10.0f} {mse_akhir:<12.6f} {'Ya' if konv else 'Tidak'}")

    plot_konvergensi_multi(
        data_plot,
        "Konvergensi XOR – Skenario 5 (Hidden Node 2 vs 3 vs 5)",
        "grafik_xor_s5.png"
    )


# ===================================================
# HELPER PRINT HEADER
# ===================================================
def header(judul, deskripsi=""):
    print(f"\n{'='*62}")
    print(f"  {judul}")
    if deskripsi:
        print(f"  {deskripsi}")
    print(f"{'='*62}")


# ===================================================
# MAIN
# ===================================================
if __name__ == "__main__":
    print("=" * 62)
    print("  JARINGAN SYARAF TIRUAN - SKENARIO PERCOBAAN DOSEN")
    print("  AND, OR: Single Layer Perceptron")
    print("  XOR    : MLP Backpropagation")
    print("=" * 62)

    X_logic = [[0,0],[0,1],[1,0],[1,1]]
    t_and   = [0, 0, 0, 1]
    t_or    = [0, 1, 1, 1]
    t_xor   = [0, 1, 1, 0]

    # ── AND ──────────────────────────────────────────────────────
    skenario_slp("AND", X_logic, t_and)

    # ── OR ───────────────────────────────────────────────────────
    skenario_slp("OR", X_logic, t_or)

    # ── XOR ──────────────────────────────────────────────────────
    skenario_xor(X_logic, t_xor)

    print(f"\n{'='*62}")
    print("  SELESAI - Semua grafik disimpan sebagai file .png")
    print(f"{'='*62}")
