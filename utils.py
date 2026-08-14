import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

class plot:
    def __init__(self, fw, afw, pfw, name, folder = 'image'):
        pass

    def loss(loss_fw, loss_afw, loss_pfw, log_scale = False, color_fw = "#48a1e1", color_afw = "#ff0ea3", color_pfw = "#38d238", name = 'image/1_loss_convergence.png'):
        plt.figure(figsize=(8, 6))
        plt.plot(loss_fw, label='Standard FW', color=color_fw, linewidth=2)
        plt.plot(loss_afw, label='Away-Step FW', color=color_afw, linewidth=2)
        plt.plot(loss_pfw, label='Pairwise FW', color=color_pfw, linewidth=2)

        plt.title('Loss Convergence vs Iterations')
        plt.xlabel('Iterations')

        if log_scale:
            plt.ylabel('Loss (Log Scale)')
            plt.yscale('log')
        else:
            plt.ylabel('Loss')

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(name, dpi=300)


    #  2: Duality Gap vs Iterazioni
    def duality_gap(gap_fw, gap_afw, gap_pfw, log_scale = False, color_fw = "#48a1e1", color_afw = "#ff0ea3", color_pfw = "#38d238", name = 'image/2_duality_gap.png'):
        plt.figure(figsize=(8, 6))
        plt.plot(gap_fw, label='Standard FW', color=color_fw, linewidth=2)
        plt.plot(gap_afw, label='Away-Step FW', color=color_afw, linewidth=2)
        plt.plot(gap_pfw, label='Pairwise FW', color=color_pfw, linewidth=2)

        plt.title('Duality Gap vs Iterations')
        plt.xlabel('Iterations')

        if log_scale:
            plt.ylabel('Gap (Log Scale)')
            plt.yscale('log')
        else:
            plt.ylabel('Gap')

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(name, dpi=300)

    #  3: Sparsità (Feature Attive) vs Iterazioni
    def sparsity(spars_fw, spars_afw, spars_pfw, color_fw = "#48a1e1", color_afw = "#ff0ea3", color_pfw = "#38d238", name = 'image/3_sparsity.png'):
        plt.figure(figsize=(8, 6))
        plt.plot(spars_fw, label='Standard FW', color=color_fw, linewidth=2)
        plt.plot(spars_afw, label='Away-Step FW', color=color_afw, linewidth=2)
        plt.plot(spars_pfw, label='Pairwise FW', color=color_pfw, linewidth=2)

        plt.title('Sparsity (Active Set dimension) vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Number of Non-Zero Features')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(name, dpi=300)


    #  4: Duality Gap vs CPU Time (Efficienza)
    def efficiency(time_fw, time_afw, time_pfw, gap_fw, gap_afw, gap_pfw, log_scale = False, color_fw = "#48a1e1", color_afw = "#ff0ea3", color_pfw = "#38d238", name = 'image/4_cpu_time.png'):
        plt.figure(figsize=(8, 6))
        plt.plot(time_fw, gap_fw, label='Standard FW', color=color_fw, linewidth=2)
        plt.plot(time_afw, gap_afw, label='Away-Step FW', color=color_afw, linewidth=2)
        plt.plot(time_pfw, gap_pfw, label='Pairwise FW', color=color_pfw, linewidth=2)

        plt.title('Efficiency: Gap vs CPU Time')
        plt.xlabel('Time (seconds)')
        
        if log_scale:
            plt.ylabel('Gap (Log Scale)')
            plt.yscale('log')
        else:
            plt.ylabel('Gap')
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(name, dpi=300)


    def weight_distr(pesi_fw, pesi_afw, pesi_pfw, color_fw = "#48a1e1", color_afw = "#ff0ea3", color_pfw = "#38d238", name = 'image/5_weight_distribution.png'):
        fig_hist, axs_hist = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        # 1. Standard FW
        axs_hist[0].hist(pesi_fw, bins=30, color=color_fw, alpha=0.7, edgecolor='black')
        axs_hist[0].set_title(f'Standard FW\n({len(pesi_fw)} feature attive)')
        axs_hist[0].set_xlabel('Valore del Coefficiente')
        axs_hist[0].set_ylabel('Frequenza Assoluta')
        axs_hist[0].grid(axis='y', linestyle='--', alpha=0.7)

        # 2. Away-Step FW
        axs_hist[1].hist(pesi_afw, bins=30, color=color_afw, alpha=0.7, edgecolor='black')
        axs_hist[1].set_title(f'Away-Step FW\n({len(pesi_afw)} feature attive)')
        axs_hist[1].set_xlabel('Valore del Coefficiente')
        axs_hist[1].grid(axis='y', linestyle='--', alpha=0.7)

        # 3. Pairwise FW
        axs_hist[2].hist(pesi_pfw, bins=30, color=color_pfw, alpha=0.7, edgecolor='black')
        axs_hist[2].set_title(f'Pairwise FW\n({len(pesi_pfw)} feature attive)')
        axs_hist[2].set_xlabel('Valore del Coefficiente')
        axs_hist[2].grid(axis='y', linestyle='--', alpha=0.7)

        fig_hist.suptitle('Distribuzione dei Coefficienti Non Nulli (Magnitudo dei Pesi)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(name, dpi=300)
        plt.show()

    def mse(mse_fw, mse_afw, mse_pfw, log_scale = False, color_fw = "#48a1e1", color_afw = "#ff0ea3", color_pfw = "#38d238", name = 'image/6_mse.png'):
        plt.figure(figsize=(8, 6))
        plt.plot(mse_fw, label='Standard FW', color=color_fw, linewidth=2)
        plt.plot(mse_afw, label='Away-Step FW', color=color_afw, linewidth=2)
        plt.plot(mse_pfw, label='Pairwise FW', color=color_pfw, linewidth=2)

        plt.title('MSE vs Iterations')
        plt.xlabel('Iterations')
        
        if log_scale:
            plt.ylabel('MSE (Log Scale)')
            plt.yscale('log')
        else:
            plt.ylabel('MSE')
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(name, dpi=300) 

class read_data:
    def __init__(self):
        pass

    def json(file):
        try:
            with open(file, 'r') as file:
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
            Y = data_matrix[:, 0]  
            X = data_matrix[:, 1:]
            return X, Y
        except FileNotFoundError:
            print(f"\nERROR: file not found '{file}'.")

    def csv(file):
        df = pd.read_csv(file)
        target_col = 'y'
        Y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        return X, Y