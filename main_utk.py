from algorithm import FrankWolfeLasso, AwayStepsFrankWolfeLasso, PairwiseFrankWolfeLasso
from load_utk import load_utkface
import numpy as np
import matplotlib.pyplot as plt
from utils import plot
import os
import pandas as pd



# tau era troppo bassa quindi teniamo un valore moderatamente alto
TAU = 0.5
#TAU = 100
#ITER = 1000
# per garantire la convergenza, aumentiamo il numero di iterazioni + avere un gap piu basso
ITER = 1000
TOLERANCE = 1e-4

print("1. Loading dataset UTKFace...")

# load_utkface() gestisce internamente: campionamento casuale delle immagini,
# conversione in scala di grigi, resize, flatten in vettori di pixel, split
# train/test e standardizzazione (fit-only-on-train). A differenza di
# main_wiki.py, qui NON serve uno StandardScaler manuale nello script
# principale, perche' load_utkface() lo fa gia' al suo interno.
data = load_utkface()

X_scaled = data["X_train"]
y_scaled = data["y_train"]

print(f"X Matrix dimensions: {X_scaled.shape}")
print(f"y Vector dimensions: {y_scaled.shape}")

# aggiunta per il warning overflow
if not np.all(np.isfinite(X_scaled)):
    n_bad = np.sum(~np.isfinite(X_scaled))
    raise ValueError(
        f"X_train contains {n_bad} values that are not finite (NaN/Inf). "
        "Check the image loading in load_utkface.py "
        "(possible corrupted images or unexpected color modes)."
    )
if not np.all(np.isfinite(y_scaled)):
    n_bad = np.sum(~np.isfinite(y_scaled))
    raise ValueError(
        f"y_train contains {n_bad} values that are not finite (NaN/Inf). "
        "Check the age parsing from the file name in load_utkface.py."
    )
print("Data check: no NaN/Inf values found in X_train and y_train. OK.")
 

print("\n2. Standard Frank-Wolfe execution...")
FWL = FrankWolfeLasso(tau=TAU, max_iter=ITER, tolerance=TOLERANCE)
fw_fitted = FWL.fit(X_scaled, y_scaled)
loss_fw, gap_fw, time_fw, spars_fw, mse_fw, niter_fw = FWL.get_history()
fw_non_zero_weights = FWL.get_number_non_zero_weights()
fw_weights = FWL.get_non_zero_weights()

print("\n--- STANDARD FW RESULTS ---")
print(f"Initial loss (after 1st vertex): {loss_fw[0]:.4f}")
print(f"Final loss: {loss_fw[-1]:.4f}")
print(f"Final gap: {gap_fw[-1]:.6f}")
print(f"Selected features: {fw_non_zero_weights} out of {X_scaled.shape[1]}")

print("\n3. Away-Step Frank-Wolfe (AFW) execution...")
AFWL = AwayStepsFrankWolfeLasso(tau=TAU, max_iter=ITER, tolerance=TOLERANCE)
afw_fitted = AFWL.fit(X_scaled, y_scaled)
loss_afw, gap_afw, time_afw, spars_afw, mse_afw, niter_afw = AFWL.get_history()
afw_non_zero_weights = AFWL.get_number_non_zero_weights()
afw_weights = AFWL.get_non_zero_weights()

print("\n--- AFW RESULTS ---")
print(f"Initial loss (after 1st vertex): {loss_afw[0]:.4f}")
print(f"Final loss: {loss_afw[-1]:.4f}")
print(f"Final gap: {gap_afw[-1]:.6f}")
print(f"Selected features: {afw_non_zero_weights} out of {X_scaled.shape[1]}")

print("\n4. Pairwise Frank-Wolfe (PFW) execution...")
PFWL = PairwiseFrankWolfeLasso(tau=TAU, max_iter=ITER, tolerance=TOLERANCE)
pfw_fitted = PFWL.fit(X_scaled, y_scaled)
loss_pfw, gap_pfw, time_pfw, spars_pfw, mse_pfw, niter_pfw = PFWL.get_history()
pfw_non_zero_weights = PFWL.get_number_non_zero_weights()
pfw_weights = PFWL.get_non_zero_weights()

print("\n--- PFW RESULTS ---")
print(f"Initial loss (after 1st vertex): {loss_pfw[0]:.4f}")
print(f"Final loss: {loss_pfw[-1]:.4f}")
print(f"Final gap: {gap_pfw[-1]:.6f}")
print(f"Selected features: {pfw_non_zero_weights} out of {X_scaled.shape[1]}")

# Plotting
print("\n5. Convergence plot generation...")

output_dir = "utk_image"
os.makedirs(output_dir, exist_ok=True)

# 1. dataframe construction
# Impacchettiamo i risultati che hai già ottenuto nel formato richiesto da utils.py
# Usiamo il nome 'FW_exact' per abbinarlo ai colori corretti impostati dal tuo collega
df_fw = pd.DataFrame({
    'run': 0, 'algorithm': 'FW_exact', 'step_size': 'exact',
    'iter': niter_fw, 'time': time_fw, 'loss': loss_fw, 
    'gap': gap_fw, 'spars': spars_fw, 'mse': mse_fw
})

df_afw = pd.DataFrame({
    'run': 0, 'algorithm': 'AFW_exact', 'step_size': 'exact',
    'iter': niter_afw, 'time': time_afw, 'loss': loss_afw, 
    'gap': gap_afw, 'spars': spars_afw, 'mse': mse_afw
})

df_pfw = pd.DataFrame({
    'run': 0, 'algorithm': 'PFW_exact', 'step_size': 'exact',
    'iter': niter_pfw, 'time': time_pfw, 'loss': loss_pfw, 
    'gap': gap_pfw, 'spars': spars_pfw, 'mse': mse_pfw
})

# Merge all in a unique DataFrame
RESULTS = pd.concat([df_fw, df_afw, df_pfw], ignore_index=True)

# 2. plot inizialization
plotter = plot()
PLOT = True

# file name
name_1 = f'{output_dir}/1_loss_convergence_utk'
name_2 = f'{output_dir}/2_duality_gap_utk'
name_3 = f'{output_dir}/3_sparsity_utk'
name_4 = f'{output_dir}/4_mse_utk'
name_5 = f'{output_dir}/5_weight_distribution_utk.png'

# 3. plots
# Loss vs Iterations
plotter.loss(RESULTS, 'iter', False, name=f"{name_1}.png", plotted=PLOT)

# Duality Gap vs Iterations (Linear and Logarithmic)
plotter.duality_gap(RESULTS, 'iter', False, name=f"{name_2}.png", plotted=PLOT)
plotter.duality_gap(RESULTS, 'iter', True, name=f"{name_2}_log.png", plotted=PLOT)

# Sparsity vs Iterations
plotter.sparsity(RESULTS, 'iter', False, name=f"{name_3}.png", plotted=PLOT)

# MSE vs Iterations
#plotter.mse(RESULTS, 'iter', False, name=f"{name_4}.png", plotted=PLOT)

# Efficiency: Duality Gap vs CPU Time (Logarithmic)
plotter.duality_gap(RESULTS, 'time', True, name=f"{name_2}_time_log.png", plotted=PLOT)

# 4. weight distribution plot
color_fw = "#48a1e1"
color_afw = "#ff0ea3"
color_pfw = "#38d238"
plot.weight_distr(fw_weights, afw_weights, pfw_weights, color_fw, color_afw, color_pfw, name=name_5, plotted=PLOT, save=True)

print("Script completed successfully! Graphs have been saved in the project folder.")