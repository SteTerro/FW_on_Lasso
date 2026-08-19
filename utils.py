from pyparsing import results
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

def sort_by_time(time_data, gap_data):
    time_arr = np.array(time_data)
    gap_arr = np.array(gap_data)
    sort_indices = np.argsort(time_arr)
    return time_arr[sort_indices], gap_arr[sort_indices]

class plot:
    def __init__(self):
        pass

    def plot_setup(self, results, x_axis, y_axis, color):

        algos = results['algorithm'].unique().tolist()

        # ho rimosso questa condizione perche altrimenti non vanno i plot
        #if len(color) != len(algos):
         #   raise ValueError(f"color must be a list of the same length as the number of algorithms \nLength color: {len(color)}, number of algorithms: {len(algos)}")

        for algo in algos:
            if 'FW_diminishing' == algo:
                plt.plot(results[results['algorithm'] == algo][x_axis], results[results['algorithm'] == algo][y_axis], label='Standard FW (diminishing)', color=color[0], linewidth=1.5) # , linestyle='--')
            if 'FW_exact' == algo:
                plt.plot(results[results['algorithm'] == algo][x_axis], results[results['algorithm'] == algo][y_axis], label='Standard FW (exact)', color=color[1], linewidth=1.5)
            if 'AFW_diminishing' == algo:
                plt.plot(results[results['algorithm'] == algo][x_axis], results[results['algorithm'] == algo][y_axis], label='Away-Step FW (diminishing)', color=color[2], linewidth=1.5) #, linestyle='--')
            if 'AFW_exact' == algo:
                plt.plot(results[results['algorithm'] == algo][x_axis], results[results['algorithm'] == algo][y_axis], label='Away-Step FW (exact)', color=color[3], linewidth=1.5)
            if 'PFW_diminishing' == algo:
                plt.plot(results[results['algorithm'] == algo][x_axis], results[results['algorithm'] == algo][y_axis], label='Pairwise FW (diminishing)', color=color[4], linewidth=1.5) # , linestyle='--')
            if 'PFW_exact' == algo:
                plt.plot(results[results['algorithm'] == algo][x_axis], results[results['algorithm'] == algo][y_axis], label='Pairwise FW (exact)', color=color[5], linewidth=1.5)
        
    def loss(self, results, x_axis = 'iter', log_scale = False, color = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#CC79A7'], name = None, plotted = True, save = False):
        plt.figure(figsize=(8, 6))

        self.plot_setup(results, x_axis, 'loss', color)

        if name is None:
            name = f'image/loss_{x_axis}'

        if x_axis == 'iter':
            plt.title('Objective Value vs Iterations')
            plt.xlabel('Iterations')
        elif x_axis == 'time':
            plt.title('Objective Value vs Time')
            plt.xlabel('Time (seconds)')
        elif x_axis == 'tau':
            plt.title('Objective Value vs Tau')
            plt.xlabel('Tau')

        if log_scale:
            plt.ylabel('Objective Value (Log Scale)')
            plt.yscale('log')
        else:
            plt.ylabel('Objective Value')

        if log_scale and name is None:
            name = f'{name}_log.png'
        elif name is None:
            name = f'{name}.png'

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save:
            plt.savefig(name, dpi=300)
        if plotted:
            plt.show()
        plt.close()

    def duality_gap(self, results, x_axis = 'iter', log_scale = False, color = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#CC79A7'], name = None, plotted = True, save = False):
        plt.figure(figsize=(8, 6))

        self.plot_setup(results, x_axis, 'gap', color)

        if name is None:
            name = f'image/gap_{x_axis}'

        if x_axis == 'iter':
            plt.title('Duality Gap vs Iterations')
            plt.xlabel('Iterations')
        elif x_axis == 'time':
            plt.title('Duality Gap vs Time')
            plt.xlabel('Time (seconds)')
        elif x_axis == 'tau':
            plt.title('Duality Gap vs Tau')
            plt.xlabel('Tau')

        if log_scale:
            plt.ylabel('Duality Gap (Log Scale)')
            plt.yscale('log')
        else:
            plt.ylabel('Duality Gap')

        if log_scale and name is None:
            name = f'{name}_log.png'
        elif name is None:
            name = f'{name}.png'

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save:
            plt.savefig(name, dpi=300)
        if plotted:
            plt.show()
        plt.close()

    def sparsity(self,results, x_axis = 'iter', log_scale = False, color = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#CC79A7'], name = None, plotted = True, save = False):
        plt.figure(figsize=(8, 6))

        self.plot_setup(results, x_axis, 'spars', color)

        if name is None:
            name = f'image/sparsity_{x_axis}'

        if x_axis == 'iter':
            plt.title('Sparsity (Active Set dimension) vs Iterations')
            plt.xlabel('Iterations')
        elif x_axis == 'time':
            plt.title('Sparsity (Active Set dimension) vs Time')
            plt.xlabel('Time (seconds)')
        elif x_axis == 'tau':
            plt.title('Sparsity (Active Set dimension) vs Tau')
            plt.xlabel('Tau')

        if log_scale:
            plt.ylabel('Sparsity (Log Scale)')
            plt.yscale('log')
        else:
            plt.ylabel('Sparsity')

        if log_scale and name is None:
            name = f'{name}_log.png'
        elif name is None:
            name = f'{name}.png'

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save:
            plt.savefig(name, dpi=300)
        if plotted:
            plt.show()
        plt.close()


    def mse(self, results, x_axis = 'iter', log_scale = False, color = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#CC79A7'], name = None, plotted = True, save = False):
        plt.figure(figsize=(8, 6))

        self.plot_setup(results, x_axis, 'mse', color)

        if name is None:
            name = f'image/mse_{x_axis}'

        if x_axis == 'iter':
            plt.title('Mean Squared Error (MSE) vs Iterations')
            plt.xlabel('Iterations')
        elif x_axis == 'time':
            plt.title('Mean Squared Error (MSE) vs Time')
            plt.xlabel('Time (seconds)')
        elif x_axis == 'tau':
            plt.title('Mean Squared Error (MSE) vs Tau')
            plt.xlabel('Tau')

        if log_scale:
            plt.ylabel('MSE (Log Scale)')
            plt.yscale('log')
        else:
            plt.ylabel('MSE')

        if log_scale and name is None:
            name = f'{name}_log.png'
        elif name is None:
            name = f'{name}.png'

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save:
            plt.savefig(name, dpi=300)
        if plotted:
            plt.show()
        plt.close()


    def weight_distr(pesi_fw, pesi_afw, pesi_pfw, color_fw = "#48a1e1", color_afw = "#ff0ea3", color_pfw = "#38d238", name = None, plotted = True, save = False):
        
        if name is None:
            name = f'image/weight_distribution.png'

        fig_hist, axs_hist = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        # 1. Standard FW
        axs_hist[0].hist(pesi_fw, bins=30, color=color_fw, alpha=0.7, edgecolor='black')
        axs_hist[0].set_title(f'Standard FW\n({len(pesi_fw)} active features)')
        axs_hist[0].set_xlabel('Coefficient Value')
        axs_hist[0].set_ylabel('Absolute Frequency')
        axs_hist[0].grid(axis='y', linestyle='--', alpha=0.7)

        # 2. Away-Step FW
        axs_hist[1].hist(pesi_afw, bins=30, color=color_afw, alpha=0.7, edgecolor='black')
        axs_hist[1].set_title(f'Away-Step FW\n({len(pesi_afw)} active features)')
        axs_hist[1].set_xlabel('Coefficient Value')
        axs_hist[1].grid(axis='y', linestyle='--', alpha=0.7)

        # 3. Pairwise FW
        axs_hist[2].hist(pesi_pfw, bins=30, color=color_pfw, alpha=0.7, edgecolor='black')
        axs_hist[2].set_title(f'Pairwise FW\n({len(pesi_pfw)} active features)')
        axs_hist[2].set_xlabel('Coefficient Value')
        axs_hist[2].grid(axis='y', linestyle='--', alpha=0.7)

        fig_hist.suptitle('Distribution of Non-Zero Coefficients (Magnitude of Weights)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(name, dpi=300)
        if plotted:
            plt.show()
        plt.close()

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

class tau_plot:
    def __init__(self):
        pass

    def plot_setup(self, results, tau, y_axis, color):

        algos = results['algorithm'].unique().tolist()

        if len(color) != len(algos):
            raise ValueError(f"color must be a list of the same length as the number of algorithms \nLength color: {len(color)}, number of algorithms: {len(algos)}")

        for algo in algos:
            if 'FW_diminishing' == algo:
                plt.plot(tau,  results[results['algorithm'] == algo]['tau'], label='Standard FW (diminishing)', color=color[0], linewidth=1.5, linestyle='--')
            if 'FW_exact' == algo:
                plt.plot(tau, results[results['algorithm'] == algo]['tau'], label='Standard FW (exact)', color=color[1], linewidth=1.5)
            if 'AFW_diminishing' == algo:
                plt.plot(tau, results[results['algorithm'] == algo]['tau'], label='Away-Step FW (diminishing)', color=color[2], linewidth=1.5, linestyle='--')
            if 'AFW_exact' == algo:
                plt.plot(tau, results[results['algorithm'] == algo]['tau'], label='Away-Step FW (exact)', color=color[3], linewidth=1.5)
            if 'PFW_diminishing' == algo:
                plt.plot(tau, results[results['algorithm'] == algo]['tau'], label='Pairwise FW (diminishing)', color=color[4], linewidth=1.5, linestyle='--')
            if 'PFW_exact' == algo:
                plt.plot(tau, results[results['algorithm'] == algo]['tau'], label='Pairwise FW (exact)', color=color[5], linewidth=1.5)
        
        for i in range(len(tau)):
            
            fw_d_vals = results[results['algorithm'] == 'FW_diminishing'][y_axis].values
            fw_e_vals = results[results['algorithm'] == 'FW_exact'][y_axis].values
            afw_d_vals = results[results['algorithm'] == 'AFW_diminishing'][y_axis].values
            afw_e_vals = results[results['algorithm'] == 'AFW_exact'][y_axis].values
            pfw_d_vals = results[results['algorithm'] == 'PFW_diminishing'][y_axis].values
            pfw_e_vals = results[results['algorithm'] == 'PFW_exact'][y_axis].values

            if fw_d_vals[i] < 1e-4:
                plt.scatter(tau[i], fw_d_vals[i], color=color[0], linewidth=1.5, marker='x', s=100)
            if fw_e_vals[i] < 1e-4:
                plt.scatter(tau[i], fw_e_vals[i], color=color[1], linewidth=1.5, marker='x', s=100)
            if afw_d_vals[i] < 1e-4:
                plt.scatter(tau[i], afw_d_vals[i], color=color[2], linewidth=1.5, marker='x', s=100)
            if afw_e_vals[i] < 1e-4:
                plt.scatter(tau[i], afw_e_vals[i], color=color[3], linewidth=1.5, marker='x', s=100)
            if pfw_d_vals[i] < 1e-4:
                plt.scatter(tau[i], pfw_d_vals[i], color=color[4], linewidth=1.5, marker='x', s=100)
            if pfw_e_vals[i] < 1e-4: 
                plt.scatter(tau[i], pfw_e_vals[i], color=color[5], linewidth=1.5, marker='x', s=100)

    def loss(self, results, tau_list, log_scale = False, color = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#CC79A7'], name = None, plotted = True, save = False):
        plt.figure(figsize=(8, 6))

        self.plot_setup(results, tau_list, 'loss', color)

        if name is None:
            name = f'image/loss_tau.png'

        plt.title('Objective Value vs Tau')
        plt.xlabel('Tau')

        if log_scale:
            plt.ylabel('Objective Value (Log Scale)')
            plt.yscale('log')
        else:
            plt.ylabel('Objective Value')

        if log_scale and name is None:
            name = f'{name}_log.png'
        elif name is None:
            name = f'{name}.png'

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save:
            plt.savefig(name, dpi=300)
        if plotted:
            plt.show()
        plt.close()

    def duality_gap(self, results, tau_list, log_scale = False, color = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#CC79A7'], name = None, plotted = True, save = False):
        plt.figure(figsize=(8, 6))

        self.plot_setup(results, tau_list, 'gap', color)

        if name is None:
            name = f'image/gap_tau.png'

        plt.title('Duality Gap vs Tau')
        plt.xlabel('Tau')

        if log_scale:
            plt.ylabel('Duality Gap (Log Scale)')
            plt.yscale('log')
        else:
            plt.ylabel('Duality Gap')

        if log_scale and name is None:
            name = f'{name}_log.png'
        elif name is None:
            name = f'{name}.png'

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save:
            plt.savefig(name, dpi=300)
        if plotted:
            plt.show()
        plt.close()

    def sparsity(self,results, tau_list, log_scale = False, color = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#CC79A7'], name = None, plotted = True, save = False):
        plt.figure(figsize=(8, 6))

        self.plot_setup(results, tau_list, 'spars', color)

        if name is None:
            name = f'image/sparsity_tau.png'

        plt.title('Sparsity (Active Set dimension) vs Tau')
        plt.xlabel('Tau')

        if log_scale:
            plt.ylabel('Sparsity (Log Scale)')
            plt.yscale('log')
        else:
            plt.ylabel('Sparsity')

        if log_scale and name is None:
            name = f'{name}_log.png'
        elif name is None:
            name = f'{name}.png'

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save:
            plt.savefig(name, dpi=300)
        if plotted:
            plt.show()
        plt.close()


    def mse(self, results, tau_list, log_scale = False, color = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#CC79A7'], name = None, plotted = True, save = False):
        plt.figure(figsize=(8, 6))

        self.plot_setup(results, tau_list, 'mse', color)

        if name is None:
            name = f'image/mse_tau.png'

        plt.title('Mean Squared Error (MSE) vs Tau')
        plt.xlabel('Tau')

        if log_scale:
            plt.ylabel('MSE (Log Scale)')
            plt.yscale('log')
        else:
            plt.ylabel('MSE')

        if log_scale and name is None:
            name = f'{name}_log.png'
        elif name is None:
            name = f'{name}.png'

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save:
            plt.savefig(name, dpi=300)
        if plotted:
            plt.show()
        plt.close()
