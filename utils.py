import matplotlib.pyplot as plt

class plot:
    def __init__(self, fw, afw, pfw, name, folder = 'image'):
        pass

    def loss(loss_fw, loss_afw, loss_pfw, color_fw = "#48a1e1", color_afw = "#ff0ea3", color_pfw = "#38d238", name = 'image/1_loss_convergence.png'):
        plt.figure(figsize=(8, 6))
        plt.plot(loss_fw, label='Standard FW', color=color_fw, linewidth=2)
        plt.plot(loss_afw, label='Away-Step FW', color=color_afw, linewidth=2)
        plt.plot(loss_pfw, label='Pairwise FW', color=color_pfw, linewidth=2)

        plt.title('Loss Convergence vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('2_loss_convergence.png', dpi=300) 


    #  2: Duality Gap vs Iterazioni
    def duality_gap(gap_fw, gap_afw, gap_pfw, color_fw = "#48a1e1", color_afw = "#ff0ea3", color_pfw = "#38d238", name = 'image/2_duality_gap.png'):
        plt.figure(figsize=(8, 6))
        plt.plot(gap_fw, label='Standard FW', color=color_fw, linewidth=2)
        plt.plot(gap_afw, label='Away-Step FW', color=color_afw, linewidth=2)
        plt.plot(gap_pfw, label='Pairwise FW', color=color_pfw, linewidth=2)

        plt.title('Duality Gap vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Gap (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('2_duality_gap.png', dpi=300)


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
        plt.savefig('2_sparsity.png', dpi=300)


    #  4: Duality Gap vs CPU Time (Efficienza)
    def efficiency(time_fw, time_afw, time_pfw, gap_fw, gap_afw, gap_pfw, color_fw = "#48a1e1", color_afw = "#ff0ea3", color_pfw = "#38d238", name = 'image/4_cpu_time.png'):
        plt.figure(figsize=(8, 6))
        plt.plot(time_fw, gap_fw, label='Standard FW', color=color_fw, linewidth=2)
        plt.plot(time_afw, gap_afw, label='Away-Step FW', color=color_afw, linewidth=2)
        plt.plot(time_pfw, gap_pfw, label='Pairwise FW', color=color_pfw, linewidth=2)

        plt.title('Efficiency: Gap vs CPU Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Gap (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('2_cpu_time.png', dpi=300)


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