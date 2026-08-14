from algorithm import FrankWolfeLasso, AwayStepsFrankWolfeLasso, PairwiseFrankWolfeLasso
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import plot
from sklearn.model_selection import train_test_split

# Global variables
# TAU = 1.0
TAU = 1
ITER = 1000
TOLERANCE = 1e-4

print("1. Loading Riboflavin dataset...")

file_name = 'data/riboflavin.csv' 

df = pd.read_csv(file_name)
target_col = 'y' 

Y = df[target_col].values
X = df.drop(columns=[target_col]).values

X_temp, X_test, y_temp, y_test = train_test_split(X, Y, test_size=0.12, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(12/76), random_state=42)


print(f"Dataset dimensions: {X.shape[0]} observations, {X.shape[1]} features (genes)")

print("\n2. Standardizing X and y...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

def compute_loss(X, y, x_weights):
    residual = (X @ x_weights) - y
    return  0.5 * np.sum(residual ** 2)

# Baseline Loss
zero_weights = np.zeros(X_train.shape[1])
baseline_loss = compute_loss(X_train, y_train, zero_weights)
print(f"\n---> INITIAL LOSS (Baseline with zero weights): {baseline_loss:.4f} <---")
print(f"---> L1 Radius (Tau) set to: {TAU} <---")

print("\n3. Standard Frank-Wolfe execution...")
FWL = FrankWolfeLasso(tau=TAU, max_iter=ITER, tolerance=TOLERANCE)
fw_fitted = FWL.fit(X_train, y_train)
loss_fw, gap_fw, time_fw, spars_fw, mse_fw = FWL.get_history()
fw_non_zero_weights = FWL.get_number_non_zero_weights()
fw_weights = FWL.get_non_zero_weights()

print("\n--- STANDARD FW RESULTS ---")
print(f"Execution Time: {time_fw[-1]}")
print(f"Initial loss (after 1st vertex): {loss_fw[0]:.4f}")
print(f"Final loss: {loss_fw[-1]:.4f}")
print(f"Final gap: {gap_fw[-1]:.6f}")
print(f"Selected features: {fw_non_zero_weights} out of {X_train.shape[1]}")
print(f"MSE on Val Set: {FWL.mse_score(X_val, y_val):.4f}")

print("\n4. Away-Step Frank-Wolfe (AFW) execution...")
AFWL = AwayStepsFrankWolfeLasso(tau=TAU, max_iter=ITER, tolerance=TOLERANCE)
afw_fittet = AFWL.fit(X_train, y_train)
loss_afw, gap_afw, time_afw, spars_afw, mse_afw = AFWL.get_history()
afw_non_zero_weights = AFWL.get_number_non_zero_weights()
afw_weights = AFWL.get_non_zero_weights()

print("\n--- AFW RESULTS ---")
print(f"Execution Time: {time_afw[-1]}")
print(f"Initial loss (after 1st vertex): {loss_afw[0]:.4f}")
print(f"Final loss: {loss_afw[-1]:.4f}")
print(f"Final gap: {gap_afw[-1]:.6f}")
print(f"Selected features: {afw_non_zero_weights} out of {X_train.shape[1]}")
print(f"MSE on Val Set: {AFWL.mse_score(X_val, y_val):.4f}")

print("\n5. Pairwise Frank-Wolfe (PFW) execution...")
PFWL = PairwiseFrankWolfeLasso(tau=TAU, max_iter=ITER, tolerance=TOLERANCE)
pfw_fitted = PFWL.fit(X_train, y_train)
loss_pfw, gap_pfw, time_pfw, spars_pfw, mse_pfw = PFWL.get_history()
pfw_non_zero_weights = PFWL.get_number_non_zero_weights()
pfw_weights = PFWL.get_non_zero_weights()

print("\n--- PFW RESULTS ---")
print(f"Execution Time: {time_pfw[-1]}")
print(f"Initial loss (after 1st vertex): {loss_pfw[0]:.4f}")
print(f"Final loss: {loss_pfw[-1]:.4f}")
print(f"Final gap: {gap_pfw[-1]:.6f}")
print(f"Selected features: {pfw_non_zero_weights} out of {X_train.shape[1]}")
print(f"MSE on Val Set: {PFWL.mse_score(X_val, y_val):.4f}")

print("\n6. Convergence plot generation...")

name_1 = f'image/1_loss_convergence_FW_{ITER}_{TAU}.png'
name_2 = f'image/2_duality_gap_FW_{ITER}_{TAU}.png'
name_3 = f'image/3_sparsity_FW_{ITER}_{TAU}.png'
name_4 = f'image/4_cpu_time_FW_{ITER}_{TAU}.png'
name_5 = f'image/5_weight_distribution_FW_{ITER}_{TAU}.png'
name_6 = f'image/6_mse_FW_{ITER}_{TAU}.png'

# Colori per variante
color_fw = "#48a1e1"
color_afw = "#ff0ea3"
color_pfw = "#38d238"

plot.loss(loss_fw, loss_afw, loss_pfw, False, color_fw, color_afw, color_pfw, name_1)
plot.duality_gap(gap_fw, gap_afw, gap_pfw, False, color_fw, color_afw, color_pfw, name_2)
plot.loss(loss_fw, loss_afw, loss_pfw, True, color_fw, color_afw, color_pfw, f'{name_1}_log.png')
plot.duality_gap(gap_fw, gap_afw, gap_pfw, True, color_fw, color_afw, color_pfw, f'{name_2}_log.png')
plot.sparsity(spars_fw, spars_afw, spars_pfw, color_fw, color_afw, color_pfw, name_3)
plot.efficiency(time_fw, time_afw, time_pfw, gap_fw, gap_afw, gap_pfw, False, color_fw, color_afw, color_pfw, name_4)
plot.efficiency(time_fw, time_afw, time_pfw, gap_fw, gap_afw, gap_pfw, True, color_fw, color_afw, color_pfw, f'{name_4}_log.png')
plot.mse(mse_fw, mse_afw, mse_pfw, False, color_fw, color_afw, color_pfw, name_6)
plot.mse(mse_fw, mse_afw, mse_pfw, True, color_fw, color_afw, color_pfw, f'{name_6}_log.png')
# plot.weight_distr(fw_weights, afw, pfw, color_fw, color_afw, color_pfw, name_5)

# Display plot
plt.show()

print("Script completed successfully! Graphs have been saved in the project folder.")
