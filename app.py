# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="JST - Skenario Percobaan", layout="wide")

# ===================================================
# TEMA HIJAU PEPOHONAN
# ===================================================
st.markdown("""
<style>
.stApp { background-color: #F0FFF4; }

[data-testid="stSidebar"] { background-color: #1B4332 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label { color: #B7E4C7 !important; }

/* Teks area utama hitam */
.stApp p, .stApp span, .stApp label,
.stApp .stSlider label, .stApp .stNumberInput label,
.stApp .stSelectbox label, .stApp .stTextInput label {
    color: #1a1a1a !important;
}

/* Tabel markdown */
.stApp table, .stApp table th, .stApp table td,
.stApp thead th, .stApp tbody td {
    color: #1a1a1a !important;
    background-color: #F0FFF4 !important;
    border-color: #74C69D !important;
}
.stApp thead th { background-color: #B7E4C7 !important; font-weight: 700; }
.stApp .stMarkdown p, .stApp .stMarkdown li,
.stApp .stMarkdown strong { color: #1a1a1a !important; }

h1 { color: #1B4332 !important; font-weight: 800; }
h2 { color: #2D6A4F !important; }
h3 { color: #40916C !important; }

[data-testid="stTabs"] [role="tablist"] {
    background-color: #D8F3DC;
    border-radius: 8px 8px 0 0;
    padding: 4px 4px 0 4px; gap: 4px;
}
[data-testid="stTabs"] [role="tab"] {
    background-color: #B7E4C7; color: #1B4332 !important;
    border-radius: 6px 6px 0 0; font-weight: 600;
    border: 1px solid #74C69D; padding: 6px 18px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background-color: #2D6A4F !important; color: #F0FFF4 !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    background-color: #40916C !important; color: #F0FFF4 !important;
}

.stButton > button {
    background-color: #2D6A4F !important; color: #F0FFF4 !important;
    border: none; border-radius: 6px; font-weight: 600;
    padding: 6px 20px;
}
.stButton > button:hover {
    background-color: #1B4332 !important; color: #74C69D !important;
}

[data-testid="stDataFrame"] { border: 1px solid #74C69D !important; border-radius: 6px; }
[data-testid="stExpander"] { background-color: #D8F3DC; border: 1px solid #74C69D; border-radius: 6px; }
header[data-testid="stHeader"] { background-color: #1B4332 !important; }
</style>
""", unsafe_allow_html=True)

# ===================================================
# FUNGSI INTI
# ===================================================
def step_function(net):
    return 1 if net >= 0 else 0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def run_slp(X, t, w1, w2, b, alpha, max_epoch=1000):
    riwayat_error, log_epochs = [], []
    konvergen = False
    epoch_konvergen = None
    for epoch in range(1, max_epoch + 1):
        total_error = 0
        epoch_data = []
        for i in range(len(X)):
            x1, x2 = float(X[i][0]), float(X[i][1])
            target = int(t[i])
            net = round(w1*x1 + w2*x2 + b, 10)
            y = step_function(net)
            error = target - y
            epoch_data.append({
                "X1": int(x1), "X2": int(x2), "Target": target,
                "W1_lama": round(w1,4), "W2_lama": round(w2,4), "b_lama": round(b,4),
                "net": round(net,4), "y": y, "error": error,
            })
            if error != 0:
                w1 = w1 + alpha * error * x1
                w2 = w2 + alpha * error * x2
                b  = b  + alpha * error
                total_error += abs(error)
        # simpan bobot setelah epoch
        for row in epoch_data:
            row["W1_baru"] = round(w1,4)
            row["W2_baru"] = round(w2,4)
            row["b_baru"]  = round(b,4)
        riwayat_error.append(total_error)
        log_epochs.append({'epoch': epoch, 'data': epoch_data,
                           'w1': w1, 'w2': w2, 'b': b, 'err': total_error})
        if total_error == 0:
            konvergen = True
            epoch_konvergen = epoch
            break
    return w1, w2, b, riwayat_error, log_epochs, konvergen, epoch_konvergen

def run_mlp(X, t, hidden_nodes, w_range, alpha, max_epoch=10000, seed=None):
    rng = np.random.RandomState(seed)
    W1 = rng.uniform(-w_range, w_range, (2, hidden_nodes))
    b1 = rng.uniform(-w_range, w_range, (hidden_nodes,))
    W2 = rng.uniform(-w_range, w_range, (hidden_nodes, 1))
    b2 = rng.uniform(-w_range, w_range, (1,))
    X_arr = np.array(X, dtype=float)
    t_arr = np.array(t, dtype=float).reshape(-1, 1)
    riwayat_mse, konvergen = [], False
    for epoch in range(1, max_epoch + 1):
        net_h = X_arr @ W1 + b1
        H     = sigmoid(net_h)
        net_o = H @ W2 + b2
        Y     = sigmoid(net_o)
        diff  = t_arr - Y
        mse   = float(np.mean(diff**2))
        riwayat_mse.append(mse)
        do  = diff * sigmoid_deriv(net_o)
        dW2 = H.T @ do;  db2 = np.sum(do, axis=0)
        dh  = (do @ W2.T) * sigmoid_deriv(net_h)
        dW1 = X_arr.T @ dh; db1_d = np.sum(dh, axis=0)
        W1 += alpha * dW1;  b1 += alpha * db1_d
        W2 += alpha * dW2;  b2 += alpha * db2
        if mse < 0.01:
            konvergen = True
            break
    Y_pred = sigmoid(sigmoid(X_arr @ W1 + b1) @ W2 + b2).flatten()
    Y_bin  = (Y_pred >= 0.5).astype(int)
    akurasi = sum(Y_bin == np.array(t)) / len(t) * 100
    result_df = pd.DataFrame({
        "X1": X_arr[:,0].astype(int), "X2": X_arr[:,1].astype(int),
        "Target": t_arr.flatten().astype(int),
        "Output Sigmoid": np.round(Y_pred, 4),
        "Output Biner": Y_bin,
        "Status": ["Benar" if y==tg else "SALAH" for y,tg in zip(Y_bin, t)]
    })
    return riwayat_mse, result_df, konvergen, len(riwayat_mse), akurasi

# ===================================================
# PLOT HELPERS
# ===================================================
WARNA = ['#1B4332','#40916C','#74C69D','#52796F']

def plot_conv(ax, err, title, color='#2D6A4F', konv_ep=None):
    ax.plot(range(1, len(err)+1), err, marker='o', markersize=3, linewidth=1.5, color=color)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Error / MSE")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3, color='#74C69D')
    ax.set_facecolor('#F8FFF9'); ax.set_ylim(bottom=0)
    if konv_ep:
        ax.axvline(x=konv_ep, color='#6B4226', linestyle='--', alpha=0.7)
        ypos = max(err)*0.5 if max(err) > 0 else 0.1
        ax.annotate(f'Konvergen\nEpoch {konv_ep}', xy=(konv_ep, 0),
                    xytext=(konv_ep+0.3, ypos), fontsize=7, color='#6B4226')

def plot_scatter_db(ax, X, t, title, w1f=None, w2f=None, bf=None):
    ax.scatter(X[t==0][:,0], X[t==0][:,1], color='#6B4226', marker='x', s=100, label='0')
    ax.scatter(X[t==1][:,0], X[t==1][:,1], color='#2D6A4F', marker='o', s=100, label='1')
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.5,1.5); ax.set_ylim(-0.5,1.5)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_facecolor('#F8FFF9')
    ax.grid(True, linestyle='--', alpha=0.4, color='#74C69D')
    if w1f is not None and w2f is not None and w2f != 0:
        xv = np.array([-0.5,1.5])
        yv = -(w1f/w2f)*xv - (bf/w2f)
        ax.plot(xv, yv, color='#6B4226', linewidth=2, label='Batas Keputusan')
    ax.legend(fontsize=8)

# ===================================================
# DATASET
# ===================================================
X    = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
t_and = np.array([0,0,0,1])
t_or  = np.array([0,1,1,1])
t_xor = np.array([0,1,1,0])
LR_LIST = [0.01, 0.05, 0.1, 0.5]

# ===================================================
# UI UTAMA
# ===================================================
st.title("Dashboard JST - Skenario Percobaan Dosen")
st.caption("Single Layer Perceptron (AND, OR) dan MLP Backpropagation (XOR)")

tab_and, tab_or, tab_xor = st.tabs(["AND", "OR", "XOR"])

# ===================================================
# FUNGSI RENDER SLP (AND / OR)
# ===================================================
def render_slp(kasus, t_data, key, default_w1, default_w2, default_b):
    s1, s2, s3, s4 = st.tabs([
        "Sk.1 - Bobot Ditentukan (LR=0.1)",
        "Sk.2 - Random ±0.5 (LR=0.1)",
        "Sk.3 - Random ±1.0 (LR=0.1)",
        "Sk.4 - Bobot Tetap, Multi LR"
    ])

    # ── Skenario 1 ──────────────────────────────────────────────────
    with s1:
        st.markdown("**Bobot awal ditentukan, Learning Rate = 0.1**")
        st.caption("Isi bobot awal sesuai nilai di Excel dosen, lalu klik Jalankan.")
        c1,c2,c3,c4 = st.columns(4)
        w1 = c1.number_input("W1 awal", value=default_w1, step=0.1, format="%.4f", key=f"{key}_1_w1")
        w2 = c2.number_input("W2 awal", value=default_w2, step=0.1, format="%.4f", key=f"{key}_1_w2")
        b  = c3.number_input("b awal",  value=default_b,  step=0.1, format="%.4f", key=f"{key}_1_b")
        ep = c4.number_input("Max Epoch", value=100, step=10, key=f"{key}_1_ep")

        if st.button("Jalankan Skenario 1", key=f"{key}_1_run"):
            w1f,w2f,bf,err,log,konv,konv_ep = run_slp(X, t_data, w1, w2, b, 0.1, int(ep))
            if konv:
                st.success(f"Konvergen pada Epoch {konv_ep} | Bobot Final: W1={w1f:.4f}, W2={w2f:.4f}, b={bf:.4f}")
            else:
                st.warning(f"Belum konvergen setelah {ep} epoch | Bobot Akhir: W1={w1f:.4f}, W2={w2f:.4f}, b={bf:.4f}")
                konv_ep = None

            # Tabel semua epoch
            st.markdown("**Detail semua epoch:**")
            all_rows = []
            for ep_log in log:
                for row in ep_log['data']:
                    all_rows.append({"Epoch": ep_log['epoch'], **row})
            st.dataframe(pd.DataFrame(all_rows), use_container_width=True, height=300)

            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(4,3))
                plot_scatter_db(ax, X, t_data, f"Decision Boundary - {kasus}", w1f, w2f, bf)
                st.pyplot(fig); plt.close()
            with col_b:
                fig, ax = plt.subplots(figsize=(4,3))
                plot_conv(ax, err, f"Konvergensi {kasus} Sk.1", konv_ep=konv_ep)
                st.pyplot(fig); plt.close()

    # ── Skenario 2 ──────────────────────────────────────────────────
    with s2:
        st.markdown("**Bobot awal random [-0.5, 0.5], Learning Rate = 0.1**")
        c1,c2,c3 = st.columns(3)
        n_run = c1.number_input("Jumlah percobaan", value=3, min_value=1, max_value=10, step=1, key=f"{key}_2_nrun")
        ep2   = c2.number_input("Max Epoch", value=200, step=50, key=f"{key}_2_ep")
        seed2 = c3.number_input("Seed (0=acak)", value=42, step=1, key=f"{key}_2_seed")

        if st.button("Jalankan Skenario 2", key=f"{key}_2_run"):
            rng = np.random.RandomState(None if seed2==0 else int(seed2))
            rows, plots = [], []
            for i in range(int(n_run)):
                w1i = round(rng.uniform(-0.5,0.5), 4)
                w2i = round(rng.uniform(-0.5,0.5), 4)
                bi  = round(rng.uniform(-0.5,0.5), 4)
                w1f,w2f,bf,err,_,konv,konv_ep = run_slp(X, t_data, w1i, w2i, bi, 0.1, int(ep2))
                rows.append({"Run": i+1,
                    "W1 awal": w1i, "W2 awal": w2i, "b awal": bi,
                    "Epoch Konvergen": konv_ep if konv else f">{int(ep2)}",
                    "W1 final": round(w1f,4), "W2 final": round(w2f,4), "b final": round(bf,4),
                    "Status": "Konvergen" if konv else "Gagal"})
                plots.append((err, konv_ep, f"Run {i+1}", WARNA[i%4]))
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            fig, axes = plt.subplots(1, min(int(n_run),4), figsize=(4*min(int(n_run),4), 3))
            if int(n_run)==1: axes=[axes]
            for ax, (e,kep,lbl,col) in zip(axes, plots[:4]):
                plot_conv(ax, e, lbl, color=col, konv_ep=kep)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Skenario 3 ──────────────────────────────────────────────────
    with s3:
        st.markdown("**Bobot awal random [-1.0, 1.0], Learning Rate = 0.1**")
        c1,c2,c3 = st.columns(3)
        n_run3 = c1.number_input("Jumlah percobaan", value=3, min_value=1, max_value=10, step=1, key=f"{key}_3_nrun")
        ep3    = c2.number_input("Max Epoch", value=200, step=50, key=f"{key}_3_ep")
        seed3  = c3.number_input("Seed (0=acak)", value=42, step=1, key=f"{key}_3_seed")

        if st.button("Jalankan Skenario 3", key=f"{key}_3_run"):
            rng = np.random.RandomState(None if seed3==0 else int(seed3))
            rows, plots = [], []
            for i in range(int(n_run3)):
                w1i = round(rng.uniform(-1.0,1.0), 4)
                w2i = round(rng.uniform(-1.0,1.0), 4)
                bi  = round(rng.uniform(-1.0,1.0), 4)
                w1f,w2f,bf,err,_,konv,konv_ep = run_slp(X, t_data, w1i, w2i, bi, 0.1, int(ep3))
                rows.append({"Run": i+1,
                    "W1 awal": w1i, "W2 awal": w2i, "b awal": bi,
                    "Epoch Konvergen": konv_ep if konv else f">{int(ep3)}",
                    "W1 final": round(w1f,4), "W2 final": round(w2f,4), "b final": round(bf,4),
                    "Status": "Konvergen" if konv else "Gagal"})
                plots.append((err, konv_ep, f"Run {i+1}", WARNA[i%4]))
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            fig, axes = plt.subplots(1, min(int(n_run3),4), figsize=(4*min(int(n_run3),4), 3))
            if int(n_run3)==1: axes=[axes]
            for ax, (e,kep,lbl,col) in zip(axes, plots[:4]):
                plot_conv(ax, e, lbl, color=col, konv_ep=kep)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Skenario 4 ──────────────────────────────────────────────────
    with s4:
        st.markdown("**Bobot awal ditentukan, LR = 0.01 / 0.05 / 0.1 / 0.5 (dijalankan sekaligus)**")
        c1,c2,c3,c4 = st.columns(4)
        w1_4 = c1.number_input("W1 awal", value=default_w1, step=0.1, format="%.4f", key=f"{key}_4_w1")
        w2_4 = c2.number_input("W2 awal", value=default_w2, step=0.1, format="%.4f", key=f"{key}_4_w2")
        b_4  = c3.number_input("b awal",  value=default_b,  step=0.1, format="%.4f", key=f"{key}_4_b")
        ep4  = c4.number_input("Max Epoch", value=200, step=50, key=f"{key}_4_ep")

        if st.button("Jalankan Skenario 4", key=f"{key}_4_run"):
            rows = []
            fig, axes = plt.subplots(1, 4, figsize=(16,3))
            for idx, lr in enumerate(LR_LIST):
                w1f,w2f,bf,err,_,konv,konv_ep = run_slp(X, t_data, w1_4, w2_4, b_4, lr, int(ep4))
                plot_conv(axes[idx], err, f"LR = {lr}", color=WARNA[idx], konv_ep=konv_ep)
                rows.append({
                    "LR": lr,
                    "Epoch Konvergen": konv_ep if konv else f">{int(ep4)}",
                    "W1 final": round(w1f,4), "W2 final": round(w2f,4), "b final": round(bf,4),
                    "Status": "Konvergen" if konv else "Gagal"
                })
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ===================================================
# FUNGSI RENDER MLP XOR
# ===================================================
def render_xor():
    s1, s2, s3, s4, s5 = st.tabs([
        "Sk.1 - Bobot Ditentukan (LR=0.1)",
        "Sk.2 - Random ±0.5 (LR=0.1)",
        "Sk.3 - Random ±1.0 (LR=0.1)",
        "Sk.4 - Bobot Tetap, Multi LR",
        "Sk.5 - Hidden Node 3 & 5"
    ])

    # ── Skenario 1 XOR ──────────────────────────────────────────────
    with s1:
        st.markdown("**MLP 2-2-1, Bobot awal ditentukan, LR = 0.1**")
        st.caption("Gunakan seed yang sama tiap kali untuk mendapat bobot yang sama (deterministik).")
        c1,c2,c3 = st.columns(3)
        seed_1 = c1.number_input("Seed bobot awal", value=42, step=1, key="xor_1_seed")
        ep_1   = c2.number_input("Max Epoch", value=10000, step=1000, key="xor_1_ep")
        lr_1   = c3.number_input("Learning Rate", value=0.1, step=0.01, format="%.2f", key="xor_1_lr")

        if st.button("Jalankan Skenario 1", key="xor_1_run"):
            err, res_df, konv, n_ep, acc = run_mlp(X, t_xor, 2, 0.5, lr_1, int(ep_1), seed=int(seed_1))
            konv_ep = n_ep if konv else None
            if konv:
                st.success(f"Konvergen Epoch {konv_ep} | Akurasi: {acc:.0f}%")
            else:
                st.warning(f"Belum konvergen | Epoch terakhir: {n_ep} | Akurasi: {acc:.0f}%")
            st.dataframe(res_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(7,3))
            plot_conv(ax, err, "Konvergensi XOR Sk.1", color='#1B4332', konv_ep=konv_ep)
            st.pyplot(fig); plt.close()

    # ── Skenario 2 XOR ──────────────────────────────────────────────
    with s2:
        st.markdown("**MLP 2-2-1, Bobot random [-0.5, 0.5], LR = 0.1**")
        c1,c2,c3 = st.columns(3)
        n2    = c1.number_input("Jumlah percobaan", value=3, min_value=1, max_value=10, step=1, key="xor_2_n")
        ep2   = c2.number_input("Max Epoch", value=10000, step=1000, key="xor_2_ep")
        seed2 = c3.number_input("Seed induk (0=acak)", value=42, step=1, key="xor_2_seed")

        if st.button("Jalankan Skenario 2", key="xor_2_run"):
            rng = np.random.RandomState(None if seed2==0 else int(seed2))
            rows, plots = [], []
            for i in range(int(n2)):
                s = int(rng.randint(0,99999))
                err, res_df, konv, n_ep, acc = run_mlp(X, t_xor, 2, 0.5, 0.1, int(ep2), seed=s)
                rows.append({"Run": i+1, "Seed": s,
                    "Epoch": n_ep if konv else f">{int(ep2)}",
                    "MSE Akhir": round(err[-1],6),
                    "Akurasi": f"{acc:.0f}%",
                    "Status": "Konvergen" if konv else "Gagal"})
                plots.append((err, n_ep if konv else None, f"Run {i+1}", WARNA[i%4]))
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            fig, axes = plt.subplots(1, min(int(n2),4), figsize=(4*min(int(n2),4), 3))
            if int(n2)==1: axes=[axes]
            for ax,(e,kep,lbl,col) in zip(axes, plots[:4]):
                plot_conv(ax, e, lbl, color=col, konv_ep=kep)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Skenario 3 XOR ──────────────────────────────────────────────
    with s3:
        st.markdown("**MLP 2-2-1, Bobot random [-1.0, 1.0], LR = 0.1**")
        c1,c2,c3 = st.columns(3)
        n3    = c1.number_input("Jumlah percobaan", value=3, min_value=1, max_value=10, step=1, key="xor_3_n")
        ep3   = c2.number_input("Max Epoch", value=10000, step=1000, key="xor_3_ep")
        seed3 = c3.number_input("Seed induk (0=acak)", value=42, step=1, key="xor_3_seed")

        if st.button("Jalankan Skenario 3", key="xor_3_run"):
            rng = np.random.RandomState(None if seed3==0 else int(seed3))
            rows, plots = [], []
            for i in range(int(n3)):
                s = int(rng.randint(0,99999))
                err, res_df, konv, n_ep, acc = run_mlp(X, t_xor, 2, 1.0, 0.1, int(ep3), seed=s)
                rows.append({"Run": i+1, "Seed": s,
                    "Epoch": n_ep if konv else f">{int(ep3)}",
                    "MSE Akhir": round(err[-1],6),
                    "Akurasi": f"{acc:.0f}%",
                    "Status": "Konvergen" if konv else "Gagal"})
                plots.append((err, n_ep if konv else None, f"Run {i+1}", WARNA[i%4]))
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            fig, axes = plt.subplots(1, min(int(n3),4), figsize=(4*min(int(n3),4), 3))
            if int(n3)==1: axes=[axes]
            for ax,(e,kep,lbl,col) in zip(axes, plots[:4]):
                plot_conv(ax, e, lbl, color=col, konv_ep=kep)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Skenario 4 XOR ──────────────────────────────────────────────
    with s4:
        st.markdown("**MLP 2-2-1, Bobot ditentukan, LR = 0.01 / 0.05 / 0.1 / 0.5**")
        c1,c2 = st.columns(2)
        ep4   = c1.number_input("Max Epoch", value=10000, step=1000, key="xor_4_ep")
        seed4 = c2.number_input("Seed bobot awal", value=42, step=1, key="xor_4_seed")

        if st.button("Jalankan Skenario 4", key="xor_4_run"):
            rows = []
            fig, axes = plt.subplots(1, 4, figsize=(16,3))
            for idx, lr in enumerate(LR_LIST):
                err, res_df, konv, n_ep, acc = run_mlp(X, t_xor, 2, 0.5, lr, int(ep4), seed=int(seed4))
                konv_ep = n_ep if konv else None
                plot_conv(axes[idx], err, f"LR = {lr}", color=WARNA[idx], konv_ep=konv_ep)
                rows.append({"LR": lr,
                    "Epoch": konv_ep if konv else f">{int(ep4)}",
                    "MSE Akhir": round(err[-1],6),
                    "Akurasi": f"{acc:.0f}%",
                    "Status": "Konvergen" if konv else "Gagal"})
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Skenario 5 XOR ──────────────────────────────────────────────
    with s5:
        st.markdown("**Bobot random [-0.5, 0.5], LR = 0.1 — Perbandingan Hidden Node 2 vs 3 vs 5**")
        c1,c2 = st.columns(2)
        ep5   = c1.number_input("Max Epoch", value=10000, step=1000, key="xor_5_ep")
        seed5 = c2.number_input("Seed (0=acak)", value=42, step=1, key="xor_5_seed")

        if st.button("Jalankan Skenario 5", key="xor_5_run"):
            hconfigs = [(2,"2-2-1",'#1B4332'), (3,"2-3-1",'#40916C'), (5,"2-5-1",'#52796F')]
            rows = []
            fig, axes = plt.subplots(1, 3, figsize=(15,3))
            detail_tabs = st.tabs(["Arsitektur 2-2-1","Arsitektur 2-3-1","Arsitektur 2-5-1"])
            for idx, (hn, lbl, col) in enumerate(hconfigs):
                s = None if seed5==0 else int(seed5)
                err, res_df, konv, n_ep, acc = run_mlp(X, t_xor, hn, 0.5, 0.1, int(ep5), seed=s)
                konv_ep = n_ep if konv else None
                plot_conv(axes[idx], err, lbl, color=col, konv_ep=konv_ep)
                rows.append({"Arsitektur": lbl, "Hidden Node": hn,
                    "Epoch": konv_ep if konv else f">{int(ep5)}",
                    "MSE Akhir": round(err[-1],6),
                    "Akurasi": f"{acc:.0f}%",
                    "Status": "Konvergen" if konv else "Gagal"})
                with detail_tabs[idx]:
                    if konv:
                        st.success(f"Konvergen Epoch {konv_ep} | Akurasi: {acc:.0f}%")
                    else:
                        st.warning(f"Belum konvergen | Akurasi: {acc:.0f}%")
                    st.dataframe(res_df, use_container_width=True)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown("**Perbandingan ketiga arsitektur:**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ===================================================
# RENDER TABS
# ===================================================
with tab_and:
    st.subheader("Kasus AND - Single Layer Perceptron")
    st.markdown("""
| X1 | X2 | Target |
|:--:|:--:|:------:|
| 0  | 0  | 0      |
| 0  | 1  | 0      |
| 1  | 0  | 0      |
| 1  | 1  | **1**  |
""")
    render_slp("AND", t_and, "and", default_w1=0.2, default_w2=0.2, default_b=-0.1)

with tab_or:
    st.subheader("Kasus OR - Single Layer Perceptron")
    st.markdown("""
| X1 | X2 | Target |
|:--:|:--:|:------:|
| 0  | 0  | 0      |
| 0  | 1  | **1**  |
| 1  | 0  | **1**  |
| 1  | 1  | **1**  |
""")
    render_slp("OR", t_or, "or", default_w1=0.0, default_w2=0.0, default_b=-0.5)

with tab_xor:
    st.subheader("Kasus XOR - Multilayer Perceptron (Backpropagation)")
    st.markdown("""
XOR bersifat **non-linearly separable** — butuh hidden layer. Aktivasi: **sigmoid** + **backpropagation**.

| X1 | X2 | Target |
|:--:|:--:|:------:|
| 0  | 0  | 0      |
| 0  | 1  | **1**  |
| 1  | 0  | **1**  |
| 1  | 1  | 0      |
""")
    render_xor()
