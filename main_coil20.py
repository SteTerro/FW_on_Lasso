from algorithm import FrankWolfeLasso, AwayStepsFrankWolfeLasso, PairwiseFrankWolfeLasso
from load_coil20 import load_coil20
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot
import os

TAU = 1
tau_str = str(TAU).replace('.','p')
ITER = 1000
TOLERANCE = 1e-4

#RESULTS = pd.DataFrame({
 #       'run': [],
  #      'algorithm': [],
   #     'step_size': [],
    #    'iter': [],
     #   'time': [],
      #  'loss': [],
       # 'gap': [],
        #'spars': [],
        #'mse': []})

#PLOT = True

print("1. Loading dataset COIL-20...")

data = load_coil20()

X_scaled = data["X_train"]
y_scaled = data["y_train"]

print(f"X Matrix dimensions: {X_scaled.shape}")
print(f"y Vector dimensions: {y_scaled.shape}")

# data diagnostic: check NaN/Inf
if not np.all(np.isfinite(X_scaled)):
    n_bad = np.sum(~np.isfinite(X_scaled))
    raise ValueError(
        f"X_train contains {n_bad} non-finite values (NaN/Inf). "
        "Check image loading in load_coil20.py."
    )
if not np.all(np.isfinite(y_scaled)):
    n_bad = np.sum(~np.isfinite(y_scaled))
    raise ValueError(
        f"y_train contains {n_bad} non-finite values (NaN/Inf). "
        "Check angle parsing from file name in load_coil20.py."
    )
print("Data check: No NaN/Inf detected in X_train and y_train. OK.")

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


print("\n5. Costruzione DataFrame risultati...")
  
def _build_results_df(label, loss, gap, time_, spars, mse, niter):
    return pd.DataFrame({
        "algorithm": label,
        "iter": niter,
        "time": time_,
        "loss": loss,
        "gap": gap,
        "spars": spars,
        "mse": mse,
    })
 
 
# exaxt liine search
df_fw = _build_results_df("FW_exact", loss_fw, gap_fw, time_fw, spars_fw, mse_fw, niter_fw)
df_afw = _build_results_df("AFW_exact", loss_afw, gap_afw, time_afw, spars_afw, mse_afw, niter_afw)
df_pfw = _build_results_df("PFW_exact", loss_pfw, gap_pfw, time_pfw, spars_pfw, mse_pfw, niter_pfw)
 
results = pd.concat([df_fw, df_afw, df_pfw], ignore_index=True)
 

# Plotting

print("\n6. Convergence plot generation...")

output_dir = "coil_image"
os.makedirs(output_dir, exist_ok=True)

default_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#CC79A7']
 
plotter = plot()  # <- istanza della classe, necessaria per loss/duality_gap/sparsity/mse
 
plotter.loss(
    results, x_axis="iter", color=default_colors,
    name=f"{output_dir}/1_loss_convergence_coil.png", save=True, plotted=False
)
plotter.duality_gap(
    results, x_axis="iter", color=default_colors,
    name=f"{output_dir}/2_duality_gap_coil.png", save=True, plotted=False
)
plotter.sparsity(
    results, x_axis="iter", color=default_colors,
    name=f"{output_dir}/3_sparsity_coil.png", save=True, plotted=False
)
plotter.mse(
    results, x_axis="iter", color=default_colors,
    name=f"{output_dir}/4_mse_coil.png", save=True, plotted=False
)
 
# weight_distr NON ha 'self' nella sua definizione in utils.py: va chiamato
# direttamente sulla classe (non su un'istanza), altrimenti il primo
# argomento verrebbe scambiato per 'self' e causerebbe lo stesso errore
# gia' incontrato con loss()/plot_setup().
color_fw = "#48a1e1"
color_afw = "#ff0ea3"
color_pfw = "#38d238"
 
plot.weight_distr(
    fw_weights, afw_weights, pfw_weights,
    color_fw, color_afw, color_pfw,
    name=f"{output_dir}/5_weight_distribution_coil.png", save=True, plotted=False
)
 
print("Script completed successfully! Graphs have been saved in the project folder.")