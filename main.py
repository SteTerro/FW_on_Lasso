from algorithm import FrankWolfeLasso, AwayStepsFrankWolfeLasso, PairwiseFrankWolfeLasso
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import plot, read_data
from sklearn.model_selection import train_test_split
import json
import os
from ucimlrepo import fetch_ucirepo

def build_history_df(algo_name, iters, loss, gap, time, spars, mse, step_size):
    return pd.DataFrame({
        'algoritmo': algo_name,
        'step_size': step_size,
        'iter': iters,
        'time': time,
        'loss': loss,
        'gap': gap,
        'spars': spars,
        'mse': mse
    })

# Global variables
# TAU = 1.0
TAU = 0.5
ITER = 1000
TOLERANCE = 1e-4
PLOT = True
# STEP = 'exact'

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

print("1. Loading dataset...")

# file_name = 'data/slice.csv' 
file_name = 'data/riboflavin.csv' 
# file_name = 'data/wikivital_mathematics.json'

X, Y = read_data.csv(file_name)
# X, Y = read_data.json(file_name)

# 0.12 ee (12/76)
X_temp, X_test, y_temp, y_test = train_test_split(X, Y, test_size=0.15, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=42, shuffle=True) # 0.25 x 0.8 = 0.2

# X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42) # 0.25 x 0.8 = 0.2

print("\nTraining size: ", X_train.shape)
print("Validation size: ", X_val.shape)
print("Test size: ", X_test.shape)

print("\nTraining targets: ", y_train.shape)
print("Validation targets: ", y_val.shape)
print("Test targets: ", y_test.shape)

print(f"Dataset dimensions: {X.shape[0]} observations, {X.shape[1]} features (genes)")

print("\n2. Standardizing X and y...")

X_train = np.concatenate((X_train, X_val), axis=0)
y_train = np.concatenate((y_train, y_val), axis=0)

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

print("\n3.A. Standard Frank-Wolfe execution (diminishing step)...")
FWLD = FrankWolfeLasso(tau=TAU, step_size='diminishing', max_iter=ITER, tolerance=TOLERANCE)
fw_fitted_d = FWLD.fit(X_train, y_train)
loss_fw_d, gap_fw_d, time_fw_d, spars_fw_d, mse_fw_d, iter_fw_d = FWLD.get_history()
fw_non_zero_weights_d = FWLD.get_number_non_zero_weights()
fw_weights_d = FWLD.get_non_zero_weights()
fw_mse_d = FWLD.mse_score(X_test, y_test)

# print("\n--- STANDARD FW RESULTS ---")
# print(f"Execution Time: {time_fw[-1]}")
# print(f"Initial loss (after 1st vertex): {loss_fw[0]:.4f}")
# print(f"Final loss: {loss_fw[-1]:.4f}")
# print(f"Final gap: {gap_fw[-1]:.6f}")
# print(f"Selected features: {fw_non_zero_weights} out of {X_train.shape[1]}")
# print(f"MSE on Val Set: {FWL.mse_score(X_test, y_test):.4f}")
# pred = FWL.predict(X_val)
# pred = pred.reshape(-1, 1)
# pred = scaler_y.inverse_transform(pred)
# y_val_real = scaler_y.inverse_transform(y_val.reshape(-1, 1))

print("\n3.B. Standard Frank-Wolfe execution (exact step)...")
FWLE = FrankWolfeLasso(tau=TAU, step_size='exact', max_iter=ITER, tolerance=TOLERANCE)
fw_fitted_e = FWLE.fit(X_train, y_train)
loss_fw_e, gap_fw_e, time_fw_e, spars_fw_e, mse_fw_e, iter_fw_e = FWLE.get_history()
fw_non_zero_weights_e = FWLE.get_number_non_zero_weights()
fw_weights_e = FWLE.get_non_zero_weights()
fw_mse_e = FWLE.mse_score(X_test, y_test)

# print("\n--- STANDARD FW RESULTS ---")
# print(f"Execution Time: {time_fw[-1]}")
# print(f"Initial loss (after 1st vertex): {loss_fw[0]:.4f}")
# print(f"Final loss: {loss_fw[-1]:.4f}")
# print(f"Final gap: {gap_fw[-1]:.6f}")
# print(f"Selected features: {fw_non_zero_weights} out of {X_train.shape[1]}")
# print(f"MSE on Val Set: {FWL.mse_score(X_test, y_test):.4f}")

# pred = FWL.predict(X_val)
# pred = pred.reshape(-1, 1)
# pred = scaler_y.inverse_transform(pred)
# y_val_real = scaler_y.inverse_transform(y_val.reshape(-1, 1))
# # print(pred - y_val_real)

print("\n4.A. Away-Step Frank-Wolfe (AFW) execution (diminishing step)...")
AFWLD = AwayStepsFrankWolfeLasso(tau=TAU, step_size='diminishing', max_iter=ITER, tolerance=TOLERANCE)
afw_fittet_d = AFWLD.fit(X_train, y_train)
loss_afw_d, gap_afw_d, time_afw_d, spars_afw_d, mse_afw_d, iter_afw_d = AFWLD.get_history()
afw_non_zero_weights_d = AFWLD.get_number_non_zero_weights()
afw_weights_d = AFWLD.get_non_zero_weights()
afw_mse_d = AFWLD.mse_score(X_test, y_test)

# print("\n--- AFW RESULTS ---")
# print(f"Execution Time: {time_afw[-1]}")
# print(f"Initial loss (after 1st vertex): {loss_afw[0]:.4f}")
# print(f"Final loss: {loss_afw[-1]:.4f}")
# print(f"Final gap: {gap_afw[-1]:.6f}")
# print(f"Selected features: {afw_non_zero_weights} out of {X_train.shape[1]}")
# print(f"MSE on Val Set: {AFWL.mse_score(X_test, y_test):.4f}")

print("\n4.B. Away-Step Frank-Wolfe (AFW) execution (exact step)...")
AFWLE = AwayStepsFrankWolfeLasso(tau=TAU, step_size='exact', max_iter=ITER, tolerance=TOLERANCE)
afw_fittet_e = AFWLE.fit(X_train, y_train)
loss_afw_e, gap_afw_e, time_afw_e, spars_afw_e, mse_afw_e, iter_afw_e = AFWLE.get_history()
afw_non_zero_weights_e = AFWLE.get_number_non_zero_weights()
afw_weights_e = AFWLE.get_non_zero_weights()
afw_mse_e = AFWLE.mse_score(X_test, y_test)

# print("\n--- AFW RESULTS ---")
# print(f"Execution Time: {time_afw[-1]}")
# print(f"Initial loss (after 1st vertex): {loss_afw[0]:.4f}")
# print(f"Final loss: {loss_afw[-1]:.4f}")
# print(f"Final gap: {gap_afw[-1]:.6f}")
# print(f"Selected features: {afw_non_zero_weights} out of {X_train.shape[1]}")
# print(f"MSE on Val Set: {AFWL.mse_score(X_test, y_test):.4f}")

print("\n5.A Pairwise Frank-Wolfe (PFW) execution (diminishing step)...")
PFWLD = PairwiseFrankWolfeLasso(tau=TAU, step_size='diminishing', max_iter=ITER, tolerance=TOLERANCE)
pfw_fitted_d = PFWLD.fit(X_train, y_train)
loss_pfw_d, gap_pfw_d, time_pfw_d, spars_pfw_d, mse_pfw_d, iter_pfw_d = PFWLD.get_history()
pfw_non_zero_weights_d = PFWLD.get_number_non_zero_weights()
pfw_weights_d = PFWLD.get_non_zero_weights()
pfw_mse_d = PFWLD.mse_score(X_test, y_test)

# print("\n--- PFW RESULTS ---")
# print(f"Execution Time: {time_pfw[-1]}")
# print(f"Initial loss (after 1st vertex): {loss_pfw[0]:.4f}")
# print(f"Final loss: {loss_pfw[-1]:.4f}")
# print(f"Final gap: {gap_pfw[-1]:.6f}")
# print(f"Selected features: {pfw_non_zero_weights} out of {X_train.shape[1]}")
# print(f"MSE on Val Set: {PFWL.mse_score(X_test, y_test):.4f}")

print("\n5.B Pairwise Frank-Wolfe (PFW) execution (exact step)...")
PFWLE = PairwiseFrankWolfeLasso(tau=TAU, step_size='exact', max_iter=ITER, tolerance=TOLERANCE)
pfw_fitted_e = PFWLE.fit(X_train, y_train)
loss_pfw_e, gap_pfw_e, time_pfw_e, spars_pfw_e, mse_pfw_e, iter_pfw_e = PFWLE.get_history()
pfw_non_zero_weights_e = PFWLE.get_number_non_zero_weights()
pfw_weights_e = PFWLE.get_non_zero_weights()
pfw_mse_e = PFWLE.mse_score(X_test, y_test)

# print("\n--- PFW RESULTS ---")
# print(f"Execution Time: {time_pfw[-1]}")
# print(f"Initial loss (after 1st vertex): {loss_pfw[0]:.4f}")
# print(f"Final loss: {loss_pfw[-1]:.4f}")
# print(f"Final gap: {gap_pfw[-1]:.6f}")
# print(f"Selected features: {pfw_non_zero_weights} out of {X_train.shape[1]}")
# print(f"MSE on Val Set: {PFWL.mse_score(X_test, y_test):.4f}")

print("GENERAL RESULTS")
print(f"{'Algorithm':<20} {'Time':<15} {'Iter':<15} {'Final Loss':<15} {'Final Gap':<15} {'Selected Features':<15} {'MSE':<15}")
print("-" * 107)
print(f"{str('FW_exact'):<20} {time_fw_e[-1]:<15.4f} {iter_fw_e[-1]:<15.4f} {loss_fw_e[-1]:<15.4f} {gap_fw_e[-1]:<15.4f} {spars_fw_e[-1]:<15.4f} {mse_fw_e[-1]:<15.4f}")
print(f"{str('FW_diminishing'):<20} {time_fw_d[-1]:<15.4f} {iter_fw_d[-1]:<15.4f} {loss_fw_d[-1]:<15.4f} {gap_fw_d[-1]:<15.4f} {spars_fw_d[-1]:<15.4f} {mse_fw_d[-1]:<15.4f}")
print(f"{str('AFW_exact'):<20} {time_afw_e[-1]:<15.4f} {iter_afw_e[-1]:<15.4f} {loss_afw_e[-1]:<15.4f} {gap_afw_e[-1]:<15.4f} {spars_afw_e[-1]:<15.4f} {mse_afw_e[-1]:<15.4f}")
print(f"{str('AFW_diminishing'):<20} {time_afw_d[-1]:<15.4f} {iter_afw_d[-1]:<15.4f} {loss_afw_d[-1]:<15.4f} {gap_afw_d[-1]:<15.4f} {spars_afw_d[-1]:<15.4f} {mse_afw_d[-1]:<15.4f}")
print(f"{str('PFW_exact'):<20} {time_pfw_e[-1]:<15.4f} {iter_pfw_e[-1]:<15.4f} {loss_pfw_e[-1]:<15.4f} {gap_pfw_e[-1]:<15.4f} {spars_pfw_e[-1]:<15.4f} {mse_pfw_e[-1]:<15.4f}")
print(f"{str('PFW_diminishing'):<20} {time_pfw_d[-1]:<15.4f} {iter_pfw_d[-1]:<15.4f} {loss_pfw_d[-1]:<15.4f} {gap_pfw_d[-1]:<15.4f} {spars_pfw_d[-1]:<15.4f} {mse_pfw_d[-1]:<15.4f}")

print("\n6. Convergence plot generation...")

# gap_fw = gap_fw[500:800]
# gap_afw = gap_afw[500:800]
# gap_pfw = gap_pfw[500:800]

# loss_fw = loss_fw[500:800]
# loss_afw = loss_afw[500:800]
# loss_pfw = loss_pfw[500:800]

# time_fw = time_fw[500:800]
# time_afw = time_afw[500:800]
# time_pfw = time_pfw[500:800]

# spars_fw = spars_fw[500:800]
# spars_afw = spars_afw[500:800]
# spars_pfw = spars_pfw[500:800]

# iter_fw = iter_fw[500:800]
# iter_afw = iter_afw[500:800]
# iter_pfw = iter_pfw[500:800]

# plot.loss(loss_fw, loss_afw, loss_pfw, iter_fw, iter_afw, iter_pfw, False, color_fw, color_afw, color_pfw, name_1)
# plot.duality_gap(gap_fw, gap_afw, gap_pfw, iter_fw, iter_afw, iter_pfw, False, color_fw, color_afw, color_pfw, name_2)
# plot.loss(loss_fw, loss_afw, loss_pfw, iter_fw, iter_afw, iter_pfw, True, color_fw, color_afw, color_pfw, f'{name_1}_log.png')
# plot.duality_gap(gap_fw, gap_afw, gap_pfw, iter_fw, iter_afw, iter_pfw, True, color_fw, color_afw, color_pfw, f'{name_2}_log.png')
# plot.sparsity(spars_fw, spars_afw, spars_pfw, iter_fw, iter_afw, iter_pfw, color_fw, color_afw, color_pfw, name_3)
# plot.efficiency(time_fw, time_afw, time_pfw, gap_fw, gap_afw, gap_pfw, False, color_fw, color_afw, color_pfw, name_4)
plot.efficiency(time_fw_d, time_afw_d, time_pfw_d, gap_fw_d, gap_afw_d, gap_pfw_d, False, color_fw, color_afw, color_pfw, f'{name_4}.png')
plot.efficiency(time_fw_d, time_afw_d, time_pfw_d, gap_fw_d, gap_afw_d, gap_pfw_d, True, color_fw, color_afw, color_pfw, f'{name_4}_log.png')
# plot.mse(mse_fw, mse_afw, mse_pfw, False, color_fw, color_afw, color_pfw, name_6)
# plot.mse(mse_fw, mse_afw, mse_pfw, True, color_fw, color_afw, color_pfw, f'{name_6}_log.png')
# plot.weight_distr(fw_weights, afw, pfw, color_fw, color_afw, color_pfw, name_5)

if PLOT:
    plt.show()

print("\n7. Writing Results to file...")



# # 3. Create the individual DataFrames
# df_pfw = build_history_df('PFW', iter_pfw, loss_pfw, gap_pfw, time_pfw, spars_pfw, mse_pfw, STEP)

# # 4. Concatenate them vertically into a single DataFrame
# df_results = pd.concat([df_fw, df_afw, df_pfw], ignore_index=True)

# # 5. Export to CSV
# csv_filename = f'results/results_{ITER}_{TAU}.csv'
# df_results.to_csv(csv_filename, index=False)

# csv_filename = f'results/results_{ITER}_{TAU}.csv'
# file_exists = os.path.isfile(csv_filename)
# df_results.to_csv(csv_filename, mode='a', index=False, header=not file_exists)

print("Script completed successfully! Graphs have been saved in the project folder.")


