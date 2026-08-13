from algorithm import FrankWolfeLasso, AwayStepsFrankWolfeLasso, PairwiseFrankWolfeLasso
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import plot

TAU = 1.0
ITER = 1000
TOLERANCE = 1e-4

print("1. Caricamento del dataset Riboflavin...")

file_name = 'data/riboflavin.csv' 

df = pd.read_csv(file_name)
target_col = 'y' 

y = df[target_col].values
X = df.drop(columns=[target_col]).values

print(f"Dataset dimensions: {X.shape[0]} observations, {X.shape[1]} features (genes)")

print("\n2. Pre-processing in corso...")
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print("\n3. Esecuzione algoritmo Frank-Wolfe...")
FWL = FrankWolfeLasso(tau=TAU, max_iter=ITER, tolerance=TOLERANCE)
fw_fitted = FWL.fit(X_scaled, y_scaled)
loss_fw, gap_fw, time_fw, spars_fw = FWL.get_history()
numero_pesi_fw = FWL.get_number_non_zero_weights()
pesi_fw = FWL.get_non_zero_weights()

print("\n--- RISULTATI FW STANDARD ---")
print(f"Loss finale raggiunta: {loss_fw[-1]:.4f}")
print(f"Gap finale: {gap_fw[-1]:.6f}")
print(f"Feature selezionate (non nulle): {numero_pesi_fw} su {X_scaled.shape[1]}")

print("\n4. Eecuzione algoritmo Away-Step Frank-Wolfe (AFW)...")
AFWL = AwayStepsFrankWolfeLasso(tau=TAU, max_iter=ITER, tolerance=TOLERANCE)
afw_fittet = AFWL.fit(X_scaled, y_scaled)
loss_afw, gap_afw, time_afw, spars_afw = AFWL.get_history()
numero_pesi_afw = AFWL.get_number_non_zero_weights()
pesi_afw = AFWL.get_non_zero_weights()

print("\n--- RISULTATI AFW ---")
print(f"Loss iniziale (dopo il 1° vertice): {loss_afw[0]:.4f}")
print(f"Loss finale: {loss_afw[-1]:.4f}")
print(f"Gap finale: {gap_afw[-1]:.6f}")
print(f"Feature selezionate (non nulle): {numero_pesi_afw} su {X_scaled.shape[1]}")

print("\n5. Esecuzione algoritmo Pairwise Frank-Wolfe (PFW)...")
PFWL = PairwiseFrankWolfeLasso(tau=TAU, max_iter=ITER, tolerance=TOLERANCE)
pfw_fitted = PFWL.fit(X_scaled, y_scaled)
loss_pfw, gap_pfw, time_pfw, spars_pfw = PFWL.get_history()
numero_pesi_pfw = PFWL.get_number_non_zero_weights()
pesi_pfw = PFWL.get_non_zero_weights()

print("\n--- RISULTATI PFW ---")
print(f"Loss iniziale (dopo il 1° vertice): {loss_pfw[0]:.4f}")
print(f"Loss finale: {loss_pfw[-1]:.4f}")
print(f"Gap finale: {gap_pfw[-1]:.6f}")
print(f"Feature selezionate (non nulle): {numero_pesi_pfw} su {X_scaled.shape[1]}")

# ==========================================
# STEP 4: plot
# ==========================================

print("\n6. Generazione dei grafici di convergenza...")

# Creazione di una griglia 2x2

name_1 = 'ribo_image/1_loss_convergence_ribo.png'
name_2 = 'ribo_image/2_duality_gap_ribo.png'
name_3 = 'ribo_image/3_sparsity_ribo.png'
name_4 = 'ribo_image/4_cpu_time_ribo.png'
name_5 = 'ribo_image/5_weight_distribution_ribo.png'

# Colori per variante
color_fw = "#48a1e1"
color_afw = "#ff0ea3"
color_pfw = "#38d238"

plot.loss(loss_fw, loss_afw, loss_pfw, color_fw, color_afw, color_pfw, name_1)
plot.duality_gap(gap_fw, gap_afw, gap_pfw, color_fw, color_afw, color_pfw, name_2)
plot.sparsity(spars_fw, spars_afw, spars_pfw, color_fw, color_afw, color_pfw, name_3)
plot.efficiency(time_fw, time_afw, time_pfw, gap_fw, gap_afw, gap_pfw, color_fw, color_afw, color_pfw, name_4)
plot.weight_distr(pesi_fw, pesi_afw, pesi_pfw, color_fw, color_afw, color_pfw, name_5)

# Mostra tutte le finestre create a schermo contemporaneamente
plt.show()

print("Script completato con successo! I grafici sono stati salvati nella cartella del progetto.")

# altri plot interessanti:
# - plot dei pesi finali (sparse) per vedere quali feature sono state selezionate
# - plot della distribuzione dei pesi (istogramma) per vedere la sparsità e la concentrazione dei pesi
# - plot della loss e del gap in funzione del tempo di esecuzione (per confrontare le velocità di convergenza)

# --- Grafico 5: Distribuzione dei Pesi (Istogramma della Sparsità) ---
# print("Generazione dell'istogramma dei pesi...")

