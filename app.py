import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="JST Dashboard", page_icon="🧠", layout="wide")

# ===================================================
# FUNGSI AKTIVASI & UTILITAS
# ===================================================
def step_function(net):
    return 1 if net >= 0 else 0

def plot_kebenaran(X, t, title, ax):
    ax.scatter(X[t==0][:,0], X[t==0][:,1], color='red', marker='x', s=100, label='0')
    ax.scatter(X[t==1][:,0], X[t==1][:,1], color='green', marker='o', s=100, label='1')
    ax.set_title(title)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

# ===================================================
# UI DASHBOARD
# ===================================================
st.title("🧠 Dashboard Simulasi Jaringan Syaraf Tiruan")
st.markdown("""
Aplikasi interaktif ini mensimulasikan proses iterasi model **Single Layer Perceptron** untuk kasus ***AND*** dan ***OR*** (Linear Separable), 
serta **Multilayer Perceptron (MLP)** untuk kasus ***XOR*** (Non-Linear Separable).
""")

st.sidebar.header("⚙️ Pengaturan Parameter")
kasus = st.sidebar.selectbox("Pilih Kasus:", ["AND", "OR", "XOR"])

w1_init = st.sidebar.number_input("Inisiasi W1 (Bobot 1)", value=0.0, step=0.1)
w2_init = st.sidebar.number_input("Inisiasi W2 (Bobot 2)", value=0.0, step=0.1)
b_init = st.sidebar.number_input("Inisiasi b (Bias)", value=0.0, step=0.1)
alpha = st.sidebar.slider("Learning Rate (alpha)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
max_epoch = st.sidebar.slider("Max Epoch", min_value=1, max_value=100, value=20, step=1)

# Dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
if kasus == "AND":
    t = np.array([0, 0, 0, 1])
elif kasus == "OR":
    t = np.array([0, 1, 1, 1])
else:
    t = np.array([0, 1, 1, 0])

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(f"Data & Output Target: {kasus}")
    df_data = pd.DataFrame({"X1": X[:,0], "X2": X[:,1], "Target": t})
    st.dataframe(df_data, use_container_width=True)

    st.subheader("Visualisasi Sebaran Data")
    fig_data, ax_data = plt.subplots(figsize=(4, 4))
    plot_kebenaran(X, t, f"Plot {kasus}", ax_data)
    st.pyplot(fig_data)

with col2:
    if kasus in ["AND", "OR"]:
        st.subheader(f"Proses Training - Single Layer Perceptron ({kasus})")
        
        w1, w2, b = w1_init, w2_init, b_init
        riwayat_error = []
        is_konvergen = False
        
        log_epochs = []
        
        for epoch in range(1, max_epoch + 1):
            total_error = 0
            epoch_data = []
            
            for i in range(len(X)):
                x1, x2 = X[i]
                target = t[i]
                
                net = w1 * x1 + w2 * x2 + b
                y = step_function(net)
                error = target - y
                
                status_keterangan = "✓ Benar" if error == 0 else "✗ Salah"
                epoch_data.append({
                    "X1": x1, "X2": x2, "Target": target,
                    "Net": round(net, 2), "Y (Output)": y, 
                    "Error": error, "Status": status_keterangan
                })
                
                if error != 0:
                    w1 = w1 + alpha * error * x1
                    w2 = w2 + alpha * error * x2
                    b  = b  + alpha * error
                    total_error += abs(error)
                    
            riwayat_error.append(total_error)
            log_epochs.append({'epoch': epoch, 'data': epoch_data, 'w1': w1, 'w2': w2, 'b': b, 'err': total_error})
            
            if total_error == 0:
                is_konvergen = True
                break
                
        # Menampilkan progress via Selectbox
        selected_epoch_idx = st.slider("Melihat Detail per Epoch:", min_value=1, max_value=len(log_epochs), value=len(log_epochs)) - 1
        e_data = log_epochs[selected_epoch_idx]
        
        st.markdown(f"**Epoch {e_data['epoch']}** | Total Error: `{e_data['err']}`")
        st.dataframe(pd.DataFrame(e_data['data']), use_container_width=True)
        st.markdown(f"*Bobot Update:* `W1` = {e_data['w1']:.2f}, `W2` = {e_data['w2']:.2f}, `bias` = {e_data['b']:.2f}")
        
        if is_konvergen:
            st.success(f"🎉 Model Konvergen pada Epoch {len(log_epochs)}!")
        else:
            st.warning("⚠️ Model belum konvergen hingga Epoch maksimal.")

        # Plot Garis Keputusan
        st.subheader("Decision Boundary & Grafik Konvergensi")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot Keputusan
        plot_kebenaran(X, t, f"Batas Keputusan - {kasus}", ax1)
        x_val = np.array([-0.5, 1.5])
        if w2 != 0:
            y_val = -(w1/w2)*x_val - (b/w2)
            ax1.plot(x_val, y_val, color='blue', linestyle='-', linewidth=2, label='Garis Pemisah')
        ax1.legend()
        
        # Plot Error
        ax2.plot(range(1, len(riwayat_error)+1), riwayat_error, marker='o', color='purple')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Total Error")
        ax2.set_title("Grafik Konvergensi Training")
        ax2.grid(True)
        
        st.pyplot(fig)

    else:
        st.subheader("Kasus XOR: Non-Linear Separable")
        st.info("💡 Karena XOR tidak linear separable, *Single Layer Perceptron* akan selalu gagal menemui konvergensi. Coba lihat demonstrasi Multilayer Perceptron di bawah.")
        
        st.markdown("### 🔌 Multilayer Perceptron (MLP 2-2-1)")
        st.markdown("""
        **Arsitektur:** 
        - Hidden Gate 1 (H1) mensimulasikan **OR**
        - Hidden Gate 2 (H2) mensimulasikan **NAND**
        - Output Gate (Y) mensimulasikan H1 **AND** H2
        """)
        
        w11, w21, b1 = 1.0, 1.0, -0.5
        w12, w22, b2 = -1.0, -1.0, 1.5
        v1, v2, b3 = 1.0, 1.0, -1.5
        
        res = []
        for i in range(len(X)):
            x1, x2 = X[i]
            target = t[i]
            
            # Hidden
            h1 = step_function(w11*x1 + w21*x2 + b1)
            h2 = step_function(w12*x1 + w22*x2 + b2)
            # Output
            y = step_function(v1*h1 + v2*h2 + b3)
            
            status = "✓" if y == target else "✗"
            res.append({
                "X1": x1, "X2": x2, "Target": target,
                "H1 (OR)": h1, "H2 (NAND)": h2, "Y (AND)": y, "Status": status
            })
            
        st.table(pd.DataFrame(res))
        st.success("🎉 MLP berhasil mengklasifikasikan XOR dengan kombinasi layer OR, NAND, dan AND!")
