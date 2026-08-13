from algorithm import FrankWolfeLasso, AwayStepsFrankWolfeLasso, PairwiseFrankWolfeLasso
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import plot

# Global variables
TAU = 1.0
ITER = 1000
TOLERANCE = 1e-4

print("1. Loading Riboflavin dataset...")

file_name = 'data/riboflavin.csv' 

df = pd.read_csv(file_name)
target_col = 'y' 

y = df[target_col].values
X = df.drop(columns=[target_col]).values

print(f"Dataset dimensions: {X.shape[0]} observations, {X.shape[1]} features (genes)")

print("\n2. Standardizing X and y...")
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

def compute_loss(X, y, x_weights):
    residual = (X @ x_weights) - y
    return  0.5 * np.sum(residual ** 2)

# Baseline Loss
zero_weights = np.zeros(X_scaled.shape[1])
baseline_loss = compute_loss(X_scaled, y_scaled, zero_weights)
print(f"\n---> INITIAL LOSS (Baseline with zero weights): {baseline_loss:.4f} <---")
print(f"---> L1 Radius (Tau) set to: {TAU} <---")

print("\n3. Standard Frank-Wolfe execution...")
FWL = FrankWolfeLasso(tau=TAU, max_iter=ITER, tolerance=TOLERANCE)
fw_fitted = FWL.fit(X_scaled, y_scaled)
loss_fw, gap_fw, time_fw, spars_fw = FWL.get_history()
fw_non_zero_weights = FWL.get_number_non_zero_weights()
fw_weights = FWL.get_non_zero_weights()

print("\n--- STANDARD FW RESULTS ---")
print(f"Initial loss (after 1st vertex): {loss_fw[0]:.4f}")
print(f"Final loss: {loss_fw[-1]:.4f}")
print(f"Final gap: {gap_fw[-1]:.6f}")
print(f"Selected features: {fw_non_zero_weights} out of {X_scaled.shape[1]}")

print("\n4. Away-Step Frank-Wolfe (AFW) execution...")
AFWL = AwayStepsFrankWolfeLasso(tau=TAU, max_iter=ITER, tolerance=TOLERANCE)
afw_fittet = AFWL.fit(X_scaled, y_scaled)
loss_afw, gap_afw, time_afw, spars_afw = AFWL.get_history()
afw_non_zero_weights = AFWL.get_number_non_zero_weights()
afw_weights = AFWL.get_non_zero_weights()

print("\n--- AFW RESULTS ---")
print(f"Initial loss (after 1st vertex): {loss_afw[0]:.4f}")
print(f"Final loss: {loss_afw[-1]:.4f}")
print(f"Final gap: {gap_afw[-1]:.6f}")
print(f"Selected features: {afw_non_zero_weights} out of {X_scaled.shape[1]}")

print("\n5. Pairwise Frank-Wolfe (PFW) execution...")
PFWL = PairwiseFrankWolfeLasso(tau=TAU, max_iter=ITER, tolerance=TOLERANCE)
pfw_fitted = PFWL.fit(X_scaled, y_scaled)
loss_pfw, gap_pfw, time_pfw, spars_pfw = PFWL.get_history()
pfw_non_zero_weights = PFWL.get_number_non_zero_weights()
pfw_weights = PFWL.get_non_zero_weights()

print("\n--- PFW RESULTS ---")
print(f"Initial loss (after 1st vertex): {loss_pfw[0]:.4f}")
print(f"Final loss: {loss_pfw[-1]:.4f}")
print(f"Final gap: {gap_pfw[-1]:.6f}")
print(f"Selected features: {pfw_non_zero_weights} out of {X_scaled.shape[1]}")

# Plotting

print("\n6. Convergence plot generation...")

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
plot.weight_distr(fw_weights, afw_weights, pfw_weights, color_fw, color_afw, color_pfw, name_5)

# Display plot
plt.show()

print("Script completed successfully! Graphs have been saved in the project folder.")


