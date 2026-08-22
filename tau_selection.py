from utils import tau_plot
from algorithm import FrankWolfeLasso, AwayStepsFrankWolfeLasso, PairwiseFrankWolfeLasso
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import *
from sklearn.model_selection import train_test_split
import json
import os
import time
import datetime
from datetime import datetime
import random

np.random.seed(42)
random.seed(42)

def build_history_df(algo_name, iters, loss, gap, time, spars, mse, step_size):
    return pd.DataFrame({
        'algorithm': algo_name,
        'step_size': step_size,
        'iter': iters,
        'time': time,
        'loss': loss,
        'gap': gap,
        'spars': spars,
        'mse': mse
    })


# Global variables
TAU = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 3, 4, 5, 6, 8, 9, 10] 
# TAU = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 ] #, 2.0, 3.0, 4.0, 5.0, 6, 7, 8, 9, 10]
# tau_str = str(TAU).replace('.','p')
ITER = 2000000
TOLERANCE = 1e-4
# STEP = 'exact'

RUN = 0
r = 0

PLOT = True
NOW = datetime.now().strftime("%m_%d_%H_%M_%S")

RUN_label = None
results_folder = 'results/'

# file_name = 'data/slice.csv' 
# file_name = 'data/riboflavin.csv' 
file_name = 'data/wikivital_mathematics.json'

RESULTS = pd.DataFrame({
        'run': [],
        'algorithm': [],
        'step_size': [],
        'iter': [],
        'time': [],
        'loss': [],
        'gap': [],
        'spars': [],
        'mse': [],
        'tau': []})

# print(f"Starting run {NOW} (month_day_hour_minute_second)...")
# print("1. Loading dataset...")

if file_name == 'data/wikivital_mathematics.json':
    X, Y = read_data.json(file_name)
else:
    X, Y = read_data.csv(file_name)

X_temp, X_test, y_temp, y_test = train_test_split(X, Y, test_size=0.15, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=42, shuffle=True) # 0.25 x 0.8 = 0.2

# print("\nTraining size: ", X_train.shape)
# print("Validation size: ", X_val.shape)
# print("Test size: ", X_test.shape)

# print("\nTraining targets: ", y_train.shape)
# print("Validation targets: ", y_val.shape)
# print("Test targets: ", y_test.shape)

# print(f"Dataset dimensions: {X.shape[0]} observations, {X.shape[1]} features")

# print("\n2. Standardizing X and y...")

# X_train = np.concatenate((X_train, X_val), axis=0)
# y_train = np.concatenate((y_train, y_val), axis=0)

X_train = np.concatenate((X_train, X_val, X_test), axis=0)
y_train = np.concatenate((y_train, y_val, y_test), axis=0)
X_test = X_train
y_test = y_train

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

FW_d_list = []
FW_e_list = []
AFW_d_list = []
AFW_e_list = []
PFW_d_list = []
PFW_e_list = []

for T in TAU:
    tau_str = str(T).replace('.','p')
    
    def compute_loss(X, y, x_weights):
        residual = (X @ x_weights) - y
        return  0.5 * np.sum(residual ** 2)

    # Baseline Loss
    zero_weights = np.zeros(X_train.shape[1])
    baseline_loss = compute_loss(X_train, y_train, zero_weights)
    # print(f"INITIAL LOSS (Baseline with zero weights): {baseline_loss:.4f}")
    # print(f"L1 Radius (Tau) set to: {T}")

    # print("\n3.A. Standard Frank-Wolfe execution (diminishing step)...")
    # FWLD = FrankWolfeLasso(tau=T, step_size='diminishing', max_iter=ITER, tolerance=TOLERANCE)
    # fw_fitted_d = FWLD.fit(X_train, y_train)
    # loss_fw_d, gap_fw_d, time_fw_d, spars_fw_d, mse_fw_d, iter_fw_d = FWLD.get_history()
    # fw_non_zero_weights_d = FWLD.get_number_non_zero_weights()
    # fw_weights_d = FWLD.get_non_zero_weights()
    # fw_mse_d = FWLD.mse_score(X_test, y_test)

    # RESULTS = pd.concat([RESULTS,pd.DataFrame({
    #         'run': [int(r)],
    #         'algorithm': ['FW_diminishing'],
    #         'step_size': ['diminishing'],
    #         'iter': [iter_fw_d[-1]],
    #         'time': [time_fw_d[-1]],
    #         'loss': [loss_fw_d[-1]],
    #         'gap': [gap_fw_d[-1]],
    #         'spars': [spars_fw_d[-1]],
    #         'mse': [mse_fw_d[-1]],
    #         'tau': [T]})])

    # print("\n3.B. Standard Frank-Wolfe execution (exact step)...")
    FWLE = FrankWolfeLasso(tau=T, step_size='exact', max_iter=ITER, tolerance=TOLERANCE)
    fw_fitted_e = FWLE.fit(X_train, y_train)
    loss_fw_e, gap_fw_e, time_fw_e, spars_fw_e, mse_fw_e, iter_fw_e = FWLE.get_history()
    fw_non_zero_weights_e = FWLE.get_number_non_zero_weights()
    fw_weights_e = FWLE.get_non_zero_weights()
    fw_mse_e = FWLE.mse_score(X_test, y_test)

    RESULTS = pd.concat([RESULTS,pd.DataFrame({
            'run': [int(r)],
            'algorithm': ['FW_exact'],
            'step_size': ['exact'],
            'iter': [iter_fw_e[-1]],
            'time': [time_fw_e[-1]],
            'loss': [loss_fw_e[-1]],
            'gap': [gap_fw_e[-1]],
            'spars': [spars_fw_e[-1]],
            'mse': [mse_fw_e[-1]],
            'tau': [T]})])

    # print("\n4.A. Away-Step Frank-Wolfe (AFW) execution (diminishing step)...")
    # AFWLD = AwayStepsFrankWolfeLasso(tau=T, step_size='diminishing', max_iter=ITER, tolerance=TOLERANCE)
    # afw_fittet_d = AFWLD.fit(X_train, y_train)
    # loss_afw_d, gap_afw_d, time_afw_d, spars_afw_d, mse_afw_d, iter_afw_d = AFWLD.get_history()
    # afw_non_zero_weights_d = AFWLD.get_number_non_zero_weights()
    # afw_weights_d = AFWLD.get_non_zero_weights()
    # afw_mse_d = AFWLD.mse_score(X_test, y_test)

    # RESULTS = pd.concat([RESULTS,pd.DataFrame({
    #         'run': [int(r)],
    #         'algorithm': ['AFW_diminishing'],
    #         'step_size': ['diminishing'],
    #         'iter': [iter_afw_d[-1]],
    #         'time': [time_afw_d[-1]],
    #         'loss': [loss_afw_d[-1]],
    #         'gap': [gap_afw_d[-1]],
    #         'spars': [spars_afw_d[-1]],
    #         'mse': [mse_afw_d[-1]],
    #         'tau': [T]})])

    # print("\n4.B. Away-Step Frank-Wolfe (AFW) execution (exact step)...")
    AFWLE = AwayStepsFrankWolfeLasso(tau=T, step_size='exact', max_iter=ITER, tolerance=TOLERANCE)
    afw_fittet_e = AFWLE.fit(X_train, y_train)
    loss_afw_e, gap_afw_e, time_afw_e, spars_afw_e, mse_afw_e, iter_afw_e = AFWLE.get_history()
    afw_non_zero_weights_e = AFWLE.get_number_non_zero_weights()
    afw_weights_e = AFWLE.get_non_zero_weights()
    afw_mse_e = AFWLE.mse_score(X_test, y_test)

    RESULTS = pd.concat([RESULTS,pd.DataFrame({
            'run': [int(r)],
            'algorithm': ['AFW_exact'],
            'step_size': ['exact'],
            'iter': [iter_afw_e[-1]],
            'time': [time_afw_e[-1]],
            'loss': [loss_afw_e[-1]],
            'gap': [gap_afw_e[-1]],
            'spars': [spars_afw_e[-1]],
            'mse': [mse_afw_e[-1]],
            'tau': [T]})])

    # print("\n5.A Pairwise Frank-Wolfe (PFW) execution (diminishing step)...")
    # PFWLD = PairwiseFrankWolfeLasso(tau=T, step_size='diminishing', max_iter=ITER, tolerance=TOLERANCE)
    # pfw_fitted_d = PFWLD.fit(X_train, y_train)
    # loss_pfw_d, gap_pfw_d, time_pfw_d, spars_pfw_d, mse_pfw_d, iter_pfw_d = PFWLD.get_history()
    # pfw_non_zero_weights_d = PFWLD.get_number_non_zero_weights()
    # pfw_weights_d = PFWLD.get_non_zero_weights()
    # pfw_mse_d = PFWLD.mse_score(X_test, y_test)

    # RESULTS = pd.concat([RESULTS,pd.DataFrame({
    #         'run': [int(r)],
    #         'algorithm': ['PFW_diminishing'],
    #         'step_size': ['diminishing'],
    #         'iter': [iter_pfw_d[-1]],
    #         'time': [time_pfw_d[-1]],
    #         'loss': [loss_pfw_d[-1]],
    #         'gap': [gap_pfw_d[-1]],
    #         'spars': [spars_pfw_d[-1]],
    #         'mse': [mse_pfw_d[-1]],
    #         'tau': [T]})])

    # print("\n5.B Pairwise Frank-Wolfe (PFW) execution (exact step)...")
    PFWLE = PairwiseFrankWolfeLasso(tau=T, step_size='exact', max_iter=ITER, tolerance=TOLERANCE)
    pfw_fitted_e = PFWLE.fit(X_train, y_train)
    loss_pfw_e, gap_pfw_e, time_pfw_e, spars_pfw_e, mse_pfw_e, iter_pfw_e = PFWLE.get_history()
    pfw_non_zero_weights_e = PFWLE.get_number_non_zero_weights()
    pfw_weights_e = PFWLE.get_non_zero_weights()
    pfw_mse_e = PFWLE.mse_score(X_test, y_test)

    RESULTS = pd.concat([RESULTS, pd.DataFrame({
            'run': [int(r)],
            'algorithm': ['PFW_exact'],
            'step_size': ['exact'],
            'iter': [iter_pfw_e[-1]],
            'time': [time_pfw_e[-1]],
            'loss': [loss_pfw_e[-1]],
            'gap': [gap_pfw_e[-1]],
            'spars': [spars_pfw_e[-1]],
            'mse': [mse_pfw_e[-1]],
            'tau': [T]})])

    # FW_d_list.append(loss_fw_d[-1])
    FW_e_list.append(iter_fw_e[-1])
    # AFW_d_list.append(loss_afw_d[-1])
    AFW_e_list.append(iter_afw_e[-1])
    # PFW_d_list.append(loss_pfw_d[-1])
    PFW_e_list.append(iter_pfw_e[-1])

    print(f"\nGENERAL RESULTS FOR TAU = {T}")
    print(f"{'Algorithm':<20} {'Time':<15} {'Iter':<15} {'Final Loss':<15} {'Final Gap':<15} {'Selected Features':<15} {'MSE':<15}")
    print("-" * 107)
    print(f"{str('FW_exact'):<20} {time_fw_e[-1]:<15.4f} {iter_fw_e[-1]:<15.4f} {loss_fw_e[-1]:<15.4f} {gap_fw_e[-1]:<15.4f} {spars_fw_e[-1]:<15.4f} {mse_fw_e[-1]:<15.4f}")
    # print(f"{str('FW_diminishing'):<20} {time_fw_d[-1]:<15.4f} {iter_fw_d[-1]:<15.4f} {loss_fw_d[-1]:<15.4f} {gap_fw_d[-1]:<15.4f} {spars_fw_d[-1]:<15.4f} {mse_fw_d[-1]:<15.4f}")
    print(f"{str('AFW_exact'):<20} {time_afw_e[-1]:<15.4f} {iter_afw_e[-1]:<15.4f} {loss_afw_e[-1]:<15.4f} {gap_afw_e[-1]:<15.4f} {spars_afw_e[-1]:<15.4f} {mse_afw_e[-1]:<15.4f}")
    # print(f"{str('AFW_diminishing'):<20} {time_afw_d[-1]:<15.4f} {iter_afw_d[-1]:<15.4f} {loss_afw_d[-1]:<15.4f} {gap_afw_d[-1]:<15.4f} {spars_afw_d[-1]:<15.4f} {mse_afw_d[-1]:<15.4f}")
    print(f"{str('PFW_exact'):<20} {time_pfw_e[-1]:<15.4f} {iter_pfw_e[-1]:<15.4f} {loss_pfw_e[-1]:<15.4f} {gap_pfw_e[-1]:<15.4f} {spars_pfw_e[-1]:<15.4f} {mse_pfw_e[-1]:<15.4f}")
    # print(f"{str('PFW_diminishing'):<20} {time_pfw_d[-1]:<15.4f} {iter_pfw_d[-1]:<15.4f} {loss_pfw_d[-1]:<15.4f} {gap_pfw_d[-1]:<15.4f} {spars_pfw_d[-1]:<15.4f} {mse_pfw_d[-1]:<15.4f}")

    print(f"Running Algo: {RESULTS['algorithm'].unique()}")

    # print("\n6. Convergence plot generation...")

    if PLOT:
        plt.show()

    # print("\n7. Writing Results to file...")

    # if file_name == 'data/riboflavin.csv':
    #     RUN_label = f"{results_folder}Ribo_{ITER}_{tau_str}_{NOW}.csv"
    # elif file_name == 'data/wikivital_mathematics.json':
    #     RUN_label = f"{results_folder}Wiki_{ITER}_{tau_str}_{NOW}.csv"
    # else:
    #     file_name = file_name.split('/')[-1]
    #     file_name = file_name.split('.')[0]
    #     RUN_label = f"{results_folder}{file_name}_{ITER}_{tau_str}_{NOW}.csv"

    RUN_label = f"{results_folder}Ribo_tau_selection_{NOW}.csv"

    # # Dataset__Iterations_Regularization_Run
    # RESULTS.to_csv(RUN_label, mode='a', index=False, header=True)

    # print(f"Run completed successfully! Saved as {RUN_label}.")

print("\n6. Convergence plot generation...")

color = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#CC79A7']
plt.figure(figsize=(8, 6))
# plt.plot(TAU, FW_d_list, label='Standard FW (diminishing)', color=color[0], linewidth=1.5, linestyle='--')
plt.plot(TAU, FW_e_list, label='Standard FW (exact)', color=color[1], linewidth=1.5)
# plt.plot(TAU, AFW_d_list, label='Away-Step FW (diminishing)', color=color[2], linewidth=1.5, linestyle='--')
plt.plot(TAU, AFW_e_list, label='Away-Step FW (exact)', color=color[3], linewidth=1.5)
# plt.plot(TAU, PFW_d_list, label='Pairwise FW (diminishing)', color=color[4], linewidth=1.5, linestyle='--')
plt.plot(TAU, PFW_e_list, label='Pairwise FW (exact)', color=color[5], linewidth=1.5)

for i in range(len(TAU)):
    # if FW_d_list[i] < 1e-4:
    #     plt.scatter(TAU[i], FW_d_list[i], color=color[0], linewidth=1.5, marker='x', s=100)
    if FW_e_list[i] < 1e-4:
        plt.scatter(TAU[i], FW_e_list[i], color=color[1], linewidth=1.5, marker='x', s=100)
    # if AFW_d_list[i] < 1e-4:
        # plt.scatter(TAU[i], AFW_d_list[i], color=color[2], linewidth=1.5, marker='x', s=100)
    if AFW_e_list[i] < 1e-4:
        plt.scatter(TAU[i], AFW_e_list[i], color=color[3], linewidth=1.5, marker='x', s=100)
    # if PFW_d_list[i] < 1e-4:
        # plt.scatter(TAU[i], PFW_d_list[i], color=color[4], linewidth=1.5, marker='x', s=100)
    if PFW_e_list[i] < 1e-4: 
        plt.scatter(TAU[i], PFW_e_list[i], color=color[5], linewidth=1.5, marker='x', s=100)

plt.title('Iterations until convergence vs Tau')
plt.xlabel('Tau')

plt.ylabel('Iterations')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.savefig(f'image/final_gap_vs_tau.png', dpi=300)
# plt.close()

plt.figure(figsize=(8, 6))
# plt.plot(TAU, FW_d_list, label='Standard FW (diminishing)', color=color[0], linewidth=1.5, linestyle='--')
plt.plot(TAU, FW_e_list, label='Standard FW (exact)', color=color[1], linewidth=1.5)
# plt.plot(TAU, AFW_d_list, label='Away-Step FW (diminishing)', color=color[2], linewidth=1.5, linestyle='--')
plt.plot(TAU, AFW_e_list, label='Away-Step FW (exact)', color=color[3], linewidth=1.5)
# plt.plot(TAU, PFW_d_list, label='Pairwise FW (diminishing)', color=color[4], linewidth=1.5, linestyle='--')
plt.plot(TAU, PFW_e_list, label='Pairwise FW (exact)', color=color[5], linewidth=1.5)

for i in range(len(TAU)):
    # if FW_d_list[i] < 1e-4:
        # plt.scatter(TAU[i], FW_d_list[i], color=color[0], linewidth=1.5, marker='x', s=100)
    if FW_e_list[i] < 1e-4:
        plt.scatter(TAU[i], FW_e_list[i], color=color[1], linewidth=1.5, marker='x', s=100)
    # if AFW_d_list[i] < 1e-4:
        plt.scatter(TAU[i], AFW_d_list[i], color=color[2], linewidth=1.5, marker='x', s=100)
    if AFW_e_list[i] < 1e-4:
        plt.scatter(TAU[i], AFW_e_list[i], color=color[3], linewidth=1.5, marker='x', s=100)
    # if PFW_d_list[i] < 1e-4:
        # plt.scatter(TAU[i], PFW_d_list[i], color=color[4], linewidth=1.5, marker='x', s=100)
    if PFW_e_list[i] < 1e-4: 
        plt.scatter(TAU[i], PFW_e_list[i], color=color[5], linewidth=1.5, marker='x', s=100)

plt.title('Iterations until convergence vs Tau')
plt.xlabel('Tau')

plt.ylabel('Iterations (Log Scale)')
plt.yscale('log')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.savefig(f'image/final_gap_vs_tau.png', dpi=300)

# plt.figure(figsize=(8, 6))
# plt.plot(TAU, FW_e_list, label='Standard FW (exact)', color=color[1], linewidth=1.5)
# plt.plot(TAU, AFW_e_list, label='Away-Step FW (exact)', color=color[3], linewidth=1.5)
# plt.plot(TAU, PFW_e_list, label='Pairwise FW (exact)', color=color[5], linewidth=1.5)

# plt.title('Objective Value vs Tau')
# plt.xlabel('Tau')

# plt.ylabel('Objective Value')

# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()

plt.show()
# plt.close()