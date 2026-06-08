import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="JST Dashboard - Skenario Percobaan", layout="wide")

# ===================================================
# FUNGSI AKTIVASI & UTILITAS
# ===================================================
def step_function(net):
    return 1 if net >= 0 else 0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def plot_convergence(ax, error_list, title, color='purple', konvergen_epoch=None):
    ax.plot(range(1, len(error_list)+1), error_list, marker='o', markersize=3,
            linewidth=1.5, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Error")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    if konvergen_epoch:
        ax.axvline(x=konvergen_epoch, color='red', linestyle='--', alpha=0.6)
        ax.annotate(f'Konvergen\nEpoch {konvergen_epoch}',
                    xy=(konvergen_epoch, 0),
                    xytext=(konvergen_epoch + 0.3, max(error_list) * 0.5 if max(error_list) > 0 else 0.1),
                    fontsize=7, color='red')

def plot_scatter(ax, X, t, title):
    ax.scatter(X[t==0][:,0], X[t==0][:,1], color='red', marker='x', s=100, label='0')
    ax.scatter(X[t==1][:,0], X[t==1][:,1], color='green', marker='o', s=100, label='1')
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=8)

# ===================================================
# SINGLE LAYER PERCEPTRON (AND / OR)
# ===================================================
def run_slp(X, t, w1, w2, b, alpha, max_epoch=1000):
    """
    Jalankan Single Layer Perceptron.
    Kembalikan (w1_final, w2_final, b_final, riwayat_error, log_epochs, konvergen)
    """
    riwayat_error = []
    log_epochs = []
    konvergen = False

    for epoch in range(1, max_epoch + 1):
        total_error = 0
        epoch_data = []
        for i in range(len(X)):
            x1, x2 = X[i]
            target = t[i]
            net = w1 * x1 + w2 * x2 + b
            y = step_function(net)
            error = target - y
            status = "✓" if error == 0 else "✗"
            epoch_data.append({
                "X1": x1, "X2": x2, "Target": target,
                "Net": round(net, 4), "Y": y, "Error": error,
                "W1": round(w1, 4), "W2": round(w2, 4), "b": round(b, 4),
                "Status": status
            })
            if error != 0:
                w1 = w1 + alpha * error * x1
                w2 = w2 + alpha * error * x2
                b  = b  + alpha * error
                total_error += abs(error)

        riwayat_error.append(total_error)
        log_epochs.append({
            'epoch': epoch, 'data': epoch_data,
            'w1': w1, 'w2': w2, 'b': b, 'err': total_error
        })
        if total_error == 0:
            konvergen = True
            break

    return w1, w2, b, riwayat_error, log_epochs, konvergen

# ===================================================
# MLP BACKPROPAGATION (XOR) - Variabel Hidden Node
# ===================================================
def run_mlp(X, t, hidden_nodes, w_init_range, alpha, max_epoch=10000, seed=None):
    """
    MLP dengan backpropagation untuk XOR.
    Arsitektur: 2 → hidden_nodes → 1
    Aktivasi: sigmoid
    """
    rng = np.random.RandomState(seed)

    # Inisialisasi bobot
    W1 = rng.uniform(-w_init_range, w_init_range, (2, hidden_nodes))
    b1 = rng.uniform(-w_init_range, w_init_range, (hidden_nodes,))
    W2 = rng.uniform(-w_init_range, w_init_range, (hidden_nodes, 1))
    b2 = rng.uniform(-w_init_range, w_init_range, (1,))

    X_arr = np.array(X, dtype=float)
    t_arr = np.array(t, dtype=float).reshape(-1, 1)

    riwayat_mse = []
    konvergen = False

    for epoch in range(1, max_epoch + 1):
        # Forward pass
        net_h = X_arr @ W1 + b1          # (4, hidden)
        H     = sigmoid(net_h)            # (4, hidden)
        net_o = H @ W2 + b2              # (4, 1)
        Y     = sigmoid(net_o)            # (4, 1)

        # Error
        diff  = t_arr - Y                # (4, 1)
        mse   = float(np.mean(diff**2))
        riwayat_mse.append(mse)

        # Backward pass
        delta_o = diff * sigmoid_deriv(net_o)           # (4, 1)
        dW2     = H.T @ delta_o                          # (hidden, 1)
        db2     = np.sum(delta_o, axis=0)

        delta_h = (delta_o @ W2.T) * sigmoid_deriv(net_h)  # (4, hidden)
        dW1     = X_arr.T @ delta_h                          # (2, hidden)
        db1     = np.sum(delta_h, axis=0)

        # Update
        W1 += alpha * dW1
        b1 += alpha * db1
        W2 += alpha * dW2
        b2 += alpha * db2

        if mse < 0.01:
            konvergen = True
            break

    # Prediksi final
    net_h = X_arr @ W1 + b1
    H     = sigmoid(net_h)
    net_o = H @ W2 + b2
    Y_pred = sigmoid(net_o).flatten()
    Y_bin  = (Y_pred >= 0.5).astype(int)

    result_df = pd.DataFrame({
        "X1": X_arr[:,0].astype(int),
        "X2": X_arr[:,1].astype(int),
        "Target": t_arr.flatten().astype(int),
        "Output (sigmoid)": np.round(Y_pred, 4),
        "Output (biner)": Y_bin,
        "Status": ["✓" if y == tgt else "✗" for y, tgt in zip(Y_bin, t)]
    })

    akurasi = sum(Y_bin == np.array(t)) / len(t) * 100
    return riwayat_mse, result_df, konvergen, len(riwayat_mse), akurasi

# ===================================================
# DATASET
# ===================================================
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
t_and = np.array([0,0,0,1])
t_or  = np.array([0,1,1,1])
t_xor = np.array([0,1,1,0])

WARNA_LR = ['#2196F3','#4CAF50','#FF9800','#E91E63']
LR_LIST   = [0.01, 0.05, 0.1, 0.5]

# ===================================================
# UI
# ===================================================
st.title("Dashboard JST – Skenario Percobaan Dosen")
st.markdown("Simulasi lengkap **Single Layer Perceptron** (AND, OR) dan **MLP Backpropagation** (XOR) sesuai skenario percobaan.")

tab_and, tab_or, tab_xor = st.tabs(["🔵 AND", "🟢 OR", "🟠 XOR"])

# ─────────────────────────────────────────────────
# HELPER: tampilkan skenario SLP (AND / OR)
# ─────────────────────────────────────────────────
def render_slp_scenario(kasus, t_data, tab_key):
    sub1, sub2, sub3, sub4 = st.tabs([
        "① Bobot Ditentukan (vs Excel)",
        "② Random ±0.5",
        "③ Random ±1.0",
        "④ Multi Learning Rate"
    ])

    # ── Skenario 1: Bobot ditentukan, LR = 0.1 ──────────────────────
    with sub1:
        st.markdown(f"### Skenario 1 – Bobot Ditentukan, LR = 0.1")
        st.markdown("Masukkan bobot awal secara manual (seperti nilai di Excel dosen).")

        c1, c2, c3 = st.columns(3)
        w1_s1 = c1.number_input("W1 awal", value=0.2, step=0.1, key=f"{tab_key}_s1_w1")
        w2_s1 = c2.number_input("W2 awal", value=0.2, step=0.1, key=f"{tab_key}_s1_w2")
        b_s1  = c3.number_input("b awal",  value=-0.1, step=0.1, key=f"{tab_key}_s1_b")
        max_ep_s1 = st.slider("Max Epoch", 1, 200, 50, key=f"{tab_key}_s1_ep")

        if st.button("▶ Jalankan", key=f"{tab_key}_s1_run"):
            w1f, w2f, bf, err, log, konv = run_slp(X, t_data, w1_s1, w2_s1, b_s1, 0.1, max_ep_s1)
            konv_ep = len(err) if konv else None

            st.markdown(f"**Status:** {'✅ Konvergen pada Epoch ' + str(konv_ep) if konv else '⚠️ Belum konvergen'}")
            st.markdown(f"**Bobot Final:** W1={w1f:.4f}, W2={w2f:.4f}, b={bf:.4f}")

            # Tabel per epoch
            sel_ep = st.slider("Pilih Epoch:", 1, len(log), len(log), key=f"{tab_key}_s1_sel") - 1
            e = log[sel_ep]
            st.markdown(f"**Epoch {e['epoch']}** | Total Error: `{e['err']}`")
            st.dataframe(pd.DataFrame(e['data']), use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(4,3))
                plot_scatter(ax, X, t_data, f"Sebaran Data {kasus}")
                if w2f != 0:
                    xv = np.array([-0.5, 1.5])
                    yv = -(w1f/w2f)*xv - (bf/w2f)
                    ax.plot(xv, yv, color='blue', linewidth=2, label='Decision Boundary')
                    ax.legend(fontsize=7)
                st.pyplot(fig)
            with col_b:
                fig2, ax2 = plt.subplots(figsize=(4,3))
                plot_convergence(ax2, err, f"Konvergensi {kasus} – Skenario 1", konvergen_epoch=konv_ep)
                st.pyplot(fig2)

    # ── Skenario 2: Random ±0.5 ──────────────────────────────────────
    with sub2:
        st.markdown("### Skenario 2 – Bobot Random ±0.5, LR = 0.1")
        n_run_s2 = st.slider("Jumlah percobaan random:", 1, 10, 3, key=f"{tab_key}_s2_nrun")
        max_ep_s2 = st.slider("Max Epoch", 1, 500, 100, key=f"{tab_key}_s2_ep")
        seed_s2   = st.number_input("Seed (0 = acak tiap klik)", value=42, step=1, key=f"{tab_key}_s2_seed")

        if st.button("▶ Jalankan", key=f"{tab_key}_s2_run"):
            rng = np.random.RandomState(None if seed_s2 == 0 else int(seed_s2))
            cols_plot = st.columns(min(n_run_s2, 3))

            summary_rows = []
            figs_conv = []
            for run_i in range(n_run_s2):
                w1i = rng.uniform(-0.5, 0.5)
                w2i = rng.uniform(-0.5, 0.5)
                bi  = rng.uniform(-0.5, 0.5)
                w1f, w2f, bf, err, log, konv = run_slp(X, t_data, w1i, w2i, bi, 0.1, max_ep_s2)
                konv_ep = len(err) if konv else None
                summary_rows.append({
                    "Run": run_i+1,
                    "W1 awal": round(w1i,4), "W2 awal": round(w2i,4), "b awal": round(bi,4),
                    "Epoch": konv_ep if konv else f">{max_ep_s2}",
                    "W1 final": round(w1f,4), "W2 final": round(w2f,4), "b final": round(bf,4),
                    "Konvergen": "✓" if konv else "✗"
                })
                figs_conv.append((err, konv_ep, f"Run {run_i+1}"))

            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            # Plot konvergensi semua run
            fig, axes = plt.subplots(1, min(n_run_s2,3), figsize=(4*min(n_run_s2,3), 3))
            if n_run_s2 == 1:
                axes = [axes]
            for ax_i, (err_i, kep, label) in enumerate(figs_conv[:3]):
                plot_convergence(axes[ax_i], err_i, label, color=WARNA_LR[ax_i % 4], konvergen_epoch=kep)
            plt.tight_layout()
            st.pyplot(fig)

    # ── Skenario 3: Random ±1.0 ──────────────────────────────────────
    with sub3:
        st.markdown("### Skenario 3 – Bobot Random ±1.0, LR = 0.1")
        n_run_s3 = st.slider("Jumlah percobaan random:", 1, 10, 3, key=f"{tab_key}_s3_nrun")
        max_ep_s3 = st.slider("Max Epoch", 1, 500, 100, key=f"{tab_key}_s3_ep")
        seed_s3   = st.number_input("Seed (0 = acak tiap klik)", value=42, step=1, key=f"{tab_key}_s3_seed")

        if st.button("▶ Jalankan", key=f"{tab_key}_s3_run"):
            rng = np.random.RandomState(None if seed_s3 == 0 else int(seed_s3))
            summary_rows = []
            figs_conv = []
            for run_i in range(n_run_s3):
                w1i = rng.uniform(-1.0, 1.0)
                w2i = rng.uniform(-1.0, 1.0)
                bi  = rng.uniform(-1.0, 1.0)
                w1f, w2f, bf, err, log, konv = run_slp(X, t_data, w1i, w2i, bi, 0.1, max_ep_s3)
                konv_ep = len(err) if konv else None
                summary_rows.append({
                    "Run": run_i+1,
                    "W1 awal": round(w1i,4), "W2 awal": round(w2i,4), "b awal": round(bi,4),
                    "Epoch": konv_ep if konv else f">{max_ep_s3}",
                    "W1 final": round(w1f,4), "W2 final": round(w2f,4), "b final": round(bf,4),
                    "Konvergen": "✓" if konv else "✗"
                })
                figs_conv.append((err, konv_ep, f"Run {run_i+1}"))

            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            fig, axes = plt.subplots(1, min(n_run_s3,3), figsize=(4*min(n_run_s3,3), 3))
            if n_run_s3 == 1:
                axes = [axes]
            for ax_i, (err_i, kep, label) in enumerate(figs_conv[:3]):
                plot_convergence(axes[ax_i], err_i, label, color=WARNA_LR[ax_i % 4], konvergen_epoch=kep)
            plt.tight_layout()
            st.pyplot(fig)

    # ── Skenario 4: Bobot ditentukan, multi LR ──────────────────────
    with sub4:
        st.markdown("### Skenario 4 – Bobot Ditentukan, LR = 0.01 / 0.05 / 0.1 / 0.5")
        st.markdown("Bandingkan pengaruh learning rate terhadap kecepatan konvergensi.")

        c1, c2, c3 = st.columns(3)
        w1_s4 = c1.number_input("W1 awal", value=0.2, step=0.1, key=f"{tab_key}_s4_w1")
        w2_s4 = c2.number_input("W2 awal", value=0.2, step=0.1, key=f"{tab_key}_s4_w2")
        b_s4  = c3.number_input("b awal",  value=-0.1, step=0.1, key=f"{tab_key}_s4_b")
        max_ep_s4 = st.slider("Max Epoch", 1, 500, 100, key=f"{tab_key}_s4_ep")

        if st.button("▶ Jalankan", key=f"{tab_key}_s4_run"):
            fig, axes = plt.subplots(1, 4, figsize=(16, 3))
            summary_rows = []

            for idx, lr in enumerate(LR_LIST):
                w1f, w2f, bf, err, log, konv = run_slp(X, t_data, w1_s4, w2_s4, b_s4, lr, max_ep_s4)
                konv_ep = len(err) if konv else None
                plot_convergence(axes[idx], err, f"LR = {lr}", color=WARNA_LR[idx], konvergen_epoch=konv_ep)
                summary_rows.append({
                    "Learning Rate": lr,
                    "Epoch Konvergen": konv_ep if konv else f">{max_ep_s4}",
                    "W1 final": round(w1f,4), "W2 final": round(w2f,4), "b final": round(bf,4),
                    "Konvergen": "✓" if konv else "✗"
                })

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("**Ringkasan Perbandingan:**")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


# ─────────────────────────────────────────────────
# HELPER: tampilkan skenario MLP XOR
# ─────────────────────────────────────────────────
def render_mlp_scenario():
    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "① Bobot Ditentukan (vs Excel)",
        "② Random ±0.5",
        "③ Random ±1.0",
        "④ Multi Learning Rate",
        "⑤ Hidden Node 3 & 5"
    ])

    MAX_EP_DEFAULT = 10000

    # ── Skenario 1 XOR: Bobot manual ─────────────────────────────────
    with sub1:
        st.markdown("### Skenario 1 XOR – Bobot Ditentukan, LR = 0.1")
        st.markdown("MLP 2-**2**-1 (2 hidden node). Bandingkan hasil dengan Excel dosen.")

        with st.expander("⚙️ Pengaturan Bobot Awal (Hidden Layer & Output)", expanded=True):
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Hidden Node 1 (h1)**")
                w11 = st.number_input("W11 (x1→h1)", value=0.5, step=0.1, key="x1_w11")
                w21 = st.number_input("W21 (x2→h1)", value=0.5, step=0.1, key="x1_w21")
                bh1 = st.number_input("Bias h1",     value=-0.5, step=0.1, key="x1_bh1")
            with c2:
                st.markdown("**Hidden Node 2 (h2)**")
                w12 = st.number_input("W12 (x1→h2)", value=-0.5, step=0.1, key="x1_w12")
                w22 = st.number_input("W22 (x2→h2)", value=-0.5, step=0.1, key="x1_w22")
                bh2 = st.number_input("Bias h2",     value=0.5,  step=0.1, key="x1_bh2")
            c3,c4 = st.columns(2)
            with c3:
                st.markdown("**Output**")
                v1  = st.number_input("V1 (h1→y)", value=0.5, step=0.1, key="x1_v1")
                v2  = st.number_input("V2 (h2→y)", value=0.5, step=0.1, key="x1_v2")
                bo  = st.number_input("Bias y",    value=-0.5, step=0.1, key="x1_bo")
            with c4:
                lr_x1  = st.number_input("Learning Rate", value=0.1, step=0.01, key="x1_lr")
                ep_x1  = st.number_input("Max Epoch", value=MAX_EP_DEFAULT, step=1000, key="x1_ep")

        if st.button("▶ Jalankan", key="xor_s1_run"):
            # Inject manual weights
            X_arr = np.array(X, dtype=float)
            t_arr = np.array(t_xor, dtype=float).reshape(-1,1)
            W1_m  = np.array([[w11, w12],[w21, w22]])
            b1_m  = np.array([bh1, bh2])
            W2_m  = np.array([[v1],[v2]])
            b2_m  = np.array([bo])

            riwayat_mse = []
            konvergen = False
            for epoch in range(1, int(ep_x1)+1):
                nh = X_arr @ W1_m + b1_m
                H  = sigmoid(nh)
                no = H @ W2_m + b2_m
                Y  = sigmoid(no)
                diff = t_arr - Y
                mse = float(np.mean(diff**2))
                riwayat_mse.append(mse)
                do  = diff * sigmoid_deriv(no)
                dW2 = H.T @ do;  db2 = np.sum(do, axis=0)
                dh  = (do @ W2_m.T) * sigmoid_deriv(nh)
                dW1 = X_arr.T @ dh; db1 = np.sum(dh, axis=0)
                W1_m += lr_x1 * dW1; b1_m += lr_x1 * db1
                W2_m += lr_x1 * dW2; b2_m += lr_x1 * db2
                if mse < 0.01:
                    konvergen = True
                    break

            Y_pred = sigmoid(sigmoid(X_arr @ W1_m + b1_m) @ W2_m + b2_m).flatten()
            Y_bin  = (Y_pred >= 0.5).astype(int)
            akurasi = sum(Y_bin == t_xor) / 4 * 100

            st.markdown(f"**Status:** {'✅ Konvergen Epoch '+str(len(riwayat_mse)) if konvergen else '⚠️ Belum konvergen'} | **Akurasi:** {akurasi:.0f}%")

            result_df = pd.DataFrame({
                "X1": [0,0,1,1], "X2": [0,1,0,1], "Target": t_xor,
                "Output (sigmoid)": np.round(Y_pred, 4),
                "Output (biner)": Y_bin,
                "Status": ["✓" if y==tgt else "✗" for y,tgt in zip(Y_bin, t_xor)]
            })
            st.dataframe(result_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(6,3))
            plot_convergence(ax, riwayat_mse, "Konvergensi XOR – Skenario 1",
                             color='#FF5722', konvergen_epoch=len(riwayat_mse) if konvergen else None)
            st.pyplot(fig)

    # ── Skenario 2 XOR: Random ±0.5 ──────────────────────────────────
    with sub2:
        st.markdown("### Skenario 2 XOR – Bobot Random ±0.5, LR = 0.1")
        n_run2  = st.slider("Jumlah percobaan:", 1, 10, 3, key="xor_s2_nrun")
        ep_s2   = st.slider("Max Epoch", 1000, 50000, MAX_EP_DEFAULT, step=1000, key="xor_s2_ep")
        seed_s2 = st.number_input("Seed (0 = acak)", value=42, step=1, key="xor_s2_seed")

        if st.button("▶ Jalankan", key="xor_s2_run"):
            rng = np.random.RandomState(None if seed_s2 == 0 else int(seed_s2))
            summary = []
            all_err = []
            for run_i in range(n_run2):
                s = int(rng.randint(0, 99999))
                err, res_df, konv, n_ep, acc = run_mlp(X, t_xor, 2, 0.5, 0.1, ep_s2, seed=s)
                summary.append({
                    "Run": run_i+1, "Seed": s,
                    "Epoch": n_ep if konv else f">{ep_s2}",
                    "MSE Akhir": round(err[-1], 6),
                    "Akurasi": f"{acc:.0f}%",
                    "Konvergen": "✓" if konv else "✗"
                })
                all_err.append((err, n_ep if konv else None, f"Run {run_i+1}"))

            st.dataframe(pd.DataFrame(summary), use_container_width=True)
            fig, axes = plt.subplots(1, min(n_run2,3), figsize=(4*min(n_run2,3), 3))
            if n_run2 == 1: axes = [axes]
            for ax_i, (err_i, kep, label) in enumerate(all_err[:3]):
                plot_convergence(axes[ax_i], err_i, label, color=WARNA_LR[ax_i%4], konvergen_epoch=kep)
            plt.tight_layout()
            st.pyplot(fig)

    # ── Skenario 3 XOR: Random ±1.0 ──────────────────────────────────
    with sub3:
        st.markdown("### Skenario 3 XOR – Bobot Random ±1.0, LR = 0.1")
        n_run3  = st.slider("Jumlah percobaan:", 1, 10, 3, key="xor_s3_nrun")
        ep_s3   = st.slider("Max Epoch", 1000, 50000, MAX_EP_DEFAULT, step=1000, key="xor_s3_ep")
        seed_s3 = st.number_input("Seed (0 = acak)", value=42, step=1, key="xor_s3_seed")

        if st.button("▶ Jalankan", key="xor_s3_run"):
            rng = np.random.RandomState(None if seed_s3 == 0 else int(seed_s3))
            summary = []
            all_err = []
            for run_i in range(n_run3):
                s = int(rng.randint(0, 99999))
                err, res_df, konv, n_ep, acc = run_mlp(X, t_xor, 2, 1.0, 0.1, ep_s3, seed=s)
                summary.append({
                    "Run": run_i+1, "Seed": s,
                    "Epoch": n_ep if konv else f">{ep_s3}",
                    "MSE Akhir": round(err[-1], 6),
                    "Akurasi": f"{acc:.0f}%",
                    "Konvergen": "✓" if konv else "✗"
                })
                all_err.append((err, n_ep if konv else None, f"Run {run_i+1}"))

            st.dataframe(pd.DataFrame(summary), use_container_width=True)
            fig, axes = plt.subplots(1, min(n_run3,3), figsize=(4*min(n_run3,3), 3))
            if n_run3 == 1: axes = [axes]
            for ax_i, (err_i, kep, label) in enumerate(all_err[:3]):
                plot_convergence(axes[ax_i], err_i, label, color=WARNA_LR[ax_i%4], konvergen_epoch=kep)
            plt.tight_layout()
            st.pyplot(fig)

    # ── Skenario 4 XOR: Multi LR ─────────────────────────────────────
    with sub4:
        st.markdown("### Skenario 4 XOR – Bobot Ditentukan, LR = 0.01 / 0.05 / 0.1 / 0.5")
        ep_s4   = st.slider("Max Epoch", 1000, 50000, MAX_EP_DEFAULT, step=1000, key="xor_s4_ep")
        seed_s4 = st.number_input("Seed bobot awal", value=42, step=1, key="xor_s4_seed")

        if st.button("▶ Jalankan", key="xor_s4_run"):
            fig, axes = plt.subplots(1, 4, figsize=(16, 3))
            summary = []
            for idx, lr in enumerate(LR_LIST):
                err, res_df, konv, n_ep, acc = run_mlp(X, t_xor, 2, 0.5, lr, ep_s4, seed=int(seed_s4))
                konv_ep = n_ep if konv else None
                plot_convergence(axes[idx], err, f"LR = {lr}", color=WARNA_LR[idx], konvergen_epoch=konv_ep)
                summary.append({
                    "LR": lr,
                    "Epoch Konvergen": konv_ep if konv else f">{ep_s4}",
                    "MSE Akhir": round(err[-1], 6),
                    "Akurasi": f"{acc:.0f}%",
                    "Konvergen": "✓" if konv else "✗"
                })
            plt.tight_layout()
            st.pyplot(fig)
            st.dataframe(pd.DataFrame(summary), use_container_width=True)

    # ── Skenario 5 XOR: Hidden Node 3 & 5 ───────────────────────────
    with sub5:
        st.markdown("### Skenario 5 XOR – Random ±0.5, LR = 0.1, Hidden Node 3 & 5")
        st.markdown("Perbandingan arsitektur: **2-2-1** vs **2-3-1** vs **2-5-1**")
        ep_s5   = st.slider("Max Epoch", 1000, 50000, MAX_EP_DEFAULT, step=1000, key="xor_s5_ep")
        seed_s5 = st.number_input("Seed (0 = acak)", value=42, step=1, key="xor_s5_seed")

        if st.button("▶ Jalankan", key="xor_s5_run"):
            hidden_configs = [2, 3, 5]
            labels = ["2-2-1", "2-3-1", "2-5-1"]
            colors = ['#2196F3', '#4CAF50', '#FF5722']
            fig, axes = plt.subplots(1, 3, figsize=(15, 3))
            summary = []
            detail_tabs = st.tabs([f"Arsitektur {l}" for l in labels])

            for idx, (hn, label) in enumerate(zip(hidden_configs, labels)):
                s = None if seed_s5 == 0 else int(seed_s5)
                err, res_df, konv, n_ep, acc = run_mlp(X, t_xor, hn, 0.5, 0.1, ep_s5, seed=s)
                konv_ep = n_ep if konv else None
                plot_convergence(axes[idx], err, f"Hidden={hn} ({label})",
                                 color=colors[idx], konvergen_epoch=konv_ep)
                summary.append({
                    "Arsitektur": label,
                    "Hidden Nodes": hn,
                    "Epoch Konvergen": konv_ep if konv else f">{ep_s5}",
                    "MSE Akhir": round(err[-1], 6),
                    "Akurasi": f"{acc:.0f}%",
                    "Konvergen": "✓" if konv else "✗"
                })
                with detail_tabs[idx]:
                    st.markdown(f"**Arsitektur {label}** | {'✅ Konvergen Epoch '+str(konv_ep) if konv else '⚠️ Belum konvergen'}")
                    st.dataframe(res_df, use_container_width=True)

            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("**Perbandingan Arsitektur:**")
            st.dataframe(pd.DataFrame(summary), use_container_width=True)


# ===================================================
# RENDER TABS
# ===================================================
with tab_and:
    st.subheader("Kasus AND – Single Layer Perceptron")
    st.markdown("""
    | X1 | X2 | Target |
    |----|----|--------|
    | 0  | 0  | 0      |
    | 0  | 1  | 0      |
    | 1  | 0  | 0      |
    | 1  | 1  | **1**  |
    """)
    render_slp_scenario("AND", t_and, "and")

with tab_or:
    st.subheader("Kasus OR – Single Layer Perceptron")
    st.markdown("""
    | X1 | X2 | Target |
    |----|----|--------|
    | 0  | 0  | 0      |
    | 0  | 1  | **1**  |
    | 1  | 0  | **1**  |
    | 1  | 1  | **1**  |
    """)
    render_slp_scenario("OR", t_or, "or")

with tab_xor:
    st.subheader("Kasus XOR – Multilayer Perceptron (Backpropagation)")
    st.markdown("""
    XOR bersifat **non-linearly separable** → memerlukan hidden layer.  
    Menggunakan **sigmoid** sebagai fungsi aktivasi + **backpropagation**.

    | X1 | X2 | Target |
    |----|----|--------|
    | 0  | 0  | 0      |
    | 0  | 1  | **1**  |
    | 1  | 0  | **1**  |
    | 1  | 1  | 0      |
    """)
    render_mlp_scenario()
