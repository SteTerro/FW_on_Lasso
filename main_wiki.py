from algorithm import FrankWolfeLasso, AwayStepsFrankWolfeLasso, PairwiseFrankWolfeLasso
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
from utils import plot

TAU = 0.5
ITER = 40
TOLERANCE = 1e-4

print("1. Loading dataset Math Essentials...")

file_name = 'data/wikivital_mathematics.json' 

try:
    with open(file_name, 'r') as file:
        dataset_json = json.load(file)
        
    # Extract the data ignoring the "edges" key (the graph structure)
    daily_visits = []
    # Take the days in chronological order (numeric keys)
    days = sorted([k for k in dataset_json.keys() if k.isdigit()], key=int)
    
    for day in days:
        daily_visits.append(dataset_json[day]['y'])
        
    # Create the matrix: 731 days (rows) x 1068 pages (columns)
    data_matrix = np.array(daily_visits)
    
    # SETUP FOR LASSO: decide to take the first page's visits (target) 
    # using the visits of all other 1067 pages (features)
    y_raw = data_matrix[:, 0]  
    X_raw = data_matrix[:, 1:] 
    
    print(f"X Matrix dimensions: {X_raw.shape}")
    print(f"y Vector dimensions: {y_raw.shape}")
    
    print("\n2. Pre-processing")
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)

    # standardize the target variable y (mean 0, variance 1)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

except FileNotFoundError:
    print(f"\nERROR: file not found '{file_name}'.")

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

name_1 = 'wiki_image/1_loss_convergence_wiki_2.png'
name_2 = 'wiki_image/2_duality_gap_wiki_2.png'
name_3 = 'wiki_image/3_sparsity_wiki_2.png'
name_4 = 'wiki_image/4_cpu_time_wiki_2.png'
name_5 = 'wiki_image/5_weight_distribution_wiki_2.png'

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


