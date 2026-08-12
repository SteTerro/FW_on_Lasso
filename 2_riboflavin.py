# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import json
from time import time

"""
scaleta:
- librerie
- funzioni per calcolare loss e gradiente
- implementazione Frank-Wolfe classico
- implementazione Away-Step Frank-Wolfe
- implementazione Pairwise Frank-Wolfe
- caricamento dataset da file JSON locale
- plot dei risultati di convergenza (loss e gap)
"""


# ============================
# STEP 1: implemetare il lasso
# ============================

def compute_loss(X, y, x_weights):
    """
    Calcola il valore della funzione obiettivo (Least Squares).
    
    Parametri:
    X : array numpy di forma (n_samples, n_features)
    y : array numpy di forma (n_samples,)
    x_weights : array numpy di forma (n_features,) - i coefficienti correnti
    
    Ritorna:
    loss : float
    """
    # Calcolo del residuo: (X * x) - y
    residual = (X @ x_weights) - y
    
    # 0.5 * norma_L2_al_quadrato
    loss = 0.5 * np.sum(residual ** 2)
    return loss

def compute_gradient(X, y, x_weights):
    """
    Calcola il vettore gradiente della funzione obiettivo.
    
    Parametri:
    X : array numpy di forma (n_samples, n_features)
    y : array numpy di forma (n_samples,)
    x_weights : array numpy di forma (n_features,) - i coefficienti correnti
    
    Ritorna:
    grad : array numpy di forma (n_features,)
    """
    # Calcolo del residuo: (X * x) - y
    residual = (X @ x_weights) - y
    
    # Gradiente: X_trasposta moltiplicata per il residuo
    grad = X.T @ residual
    return grad

# ===================================
# STEP 2: implementare il Frank-Wolfe
# ===================================

# iniziamo con la funzione dell'oracolo (LMO)

def lmo_l1(gradient, tau):
    """
    Linear Minimization Oracle (LMO) for ball L1.
    Find the optimal atom/vertex s_t to minimize the dot product with gradient.
    """
    # s = zero vector of the same shape as gradient
    s = np.zeros_like(gradient)
    
    # Troviamo l'indice della feature col gradiente massimo in valore assoluto
    idx_max = np.argmax(np.abs(gradient))
    
    # Assegniamo il valore tau a quell'indice, con segno OPPOSTO al gradiente 
    # (per puntare nella direzione di massima discesa)
    s[idx_max] = -np.sign(gradient[idx_max]) * tau
    
    return s

# FW for lasso
def frank_wolfe_lasso(X, y, tau, max_iter=1000, tol=1e-4):
    """
    Algoritmo Frank-Wolfe classico per il problema LASSO.
    """
    n_samples, n_features = X.shape
    # Si parte sempre dall'origine (il centro della palla, massima sparsità)
    x_t = np.zeros(n_features) 

    history_loss, history_gap, history_time, history_sparsity = [], [], [], []
    t0 = time()
    
    for t in range(max_iter):
        # 1. Calcolo del gradiente corrente
        grad = compute_gradient(X, y, x_t)
        
        # 2. Chiamata all'Oracolo (trova il vertice s_t)
        s_t = lmo_l1(grad, tau)
        
        # 3. Direzione di aggiornamento
        d_t = s_t - x_t
        
        # 4. Calcolo del Frank-Wolfe Duality Gap
        # formula: <grad, x_t - s_t> che equivale a <grad, -d_t>
        gap = np.dot(grad, -d_t)
        
        # Salviamo le metriche per i plot finali
        history_loss.append(compute_loss(X, y, x_t))
        history_gap.append(gap)
        history_time.append(time() - t0)
        history_sparsity.append(np.sum(np.abs(x_t) > 1e-8))
        
        # 5. Criterio di arresto: se il gap è minore della tolleranza, fermati!
        if gap <= tol:
            print(f"Convergenza raggiunta all'iterazione {t} con gap: {gap:.6f}")
            break
            
        # 6. Exact Line Search
        # Trova il passo gamma ottimale minimizzando analiticamente la parabola
        Xd = X @ d_t
        # mattiamo 1e-10 al den per evitare divisioni per zero
        gamma_ottimale = gap / (np.sum(Xd ** 2) + 1e-10) 
        
        # Il passo di FW deve sempre essere compreso tra 0 e 1 (per restare nel dominio)
        gamma = np.clip(gamma_ottimale, 0.0, 1.0)
        
        # 7. Aggiornamento dei coefficienti
        x_t = x_t + gamma * d_t
        
    return x_t, history_loss, history_gap, history_time, history_sparsity

# problema di FW: zigzagging
# proviamo delle varianti: away step e pairwise step (fully corrective step?)

# introduciamo il concetto di active set (FW_survey.pdf 5.3 Variants)
# active set:= set of all constrains active at the current solution (i.e., the set of vertices that have been selected so far)
# avremo: 1) atoms selected so far (i.e., the set of vertices that have been selected so far)
# 2) weight of each atom (i.e., the coefficients of the selected vertices)
"""
Avendo a disposizione questo registro, il ciclo iterativo per l'Away-Step cambierà così:
- LMO Classico: Cerca il vertice migliore su tutto il dominio (esattamente come prima) per creare la direzione FW.
- Away Oracle: Guarda solo all'interno dell'Active Set, trova il vertice peggiore (quello che rema contro la discesa del gradiente) e crea una direzione "Away" per sottrargli peso.
- Scelta: L'algoritmo calcola quale delle due direzioni abbassa di più la funzione obiettivo e fa il passo in quella direzione.
"""


def away_steps_fw_lasso(X, y, tau, max_iters=1000, tolerance=1e-4):
    n_samples, n_features = X.shape
    x = np.zeros(n_features)

    grad_0 = compute_gradient(X, y, x)
    start_idx = np.argmax(np.abs(grad_0))
    start_sign = -np.sign(grad_0[start_idx])

    x[start_idx] = start_sign * tau

    # Active Set a dizionario (indice, segno)
    weights = {(start_idx, start_sign): 1.0}

    #obj_history = []
    #gap_history = []
    history_loss, history_gap, history_time, history_sparsity = [], [], [], []
    t0 = time()

    for k in range(max_iters):
        grad = compute_gradient(X, y, x)
        history_loss.append(compute_loss(X, y, x))

        # FW VERTEX
        s_idx = np.argmax(np.abs(grad))
        s_sign = -np.sign(grad[s_idx])
        s_vec = np.zeros(n_features)
        s_vec[s_idx] = s_sign * tau

        # AWAY VERTEX 
        max_val = -np.inf
        v_key = None
        for key in weights.keys():
            idx, sign = key
            val = grad[idx] * sign * tau
            if val > max_val:
                max_val = val
                v_key = key

        v_vec = np.zeros(n_features)
        v_idx, v_sign = v_key
        v_vec[v_idx] = v_sign * tau

        # STOPPING CONDITION
        fw_gap = -np.dot(grad, s_vec - x)

        history_loss.append(compute_loss(X, y, x))
        history_gap.append(fw_gap)
        history_time.append(time() - t0)            
        # history_sparsity.append(np.sum(np.abs(x) > 1e-8))
        history_sparsity.append(len(weights))

        if fw_gap <= tolerance:
            print(f"AFW Convergenza raggiunta all'iterazione {k} con gap: {fw_gap:.6f}")
            break

        # DIREZIONE E PROTEZIONE OVERFLOW
        away_gap = -np.dot(grad, x - v_vec)

        # Protezione: se c'è un solo vertice, o se il FW gap è maggiore, passo FW
        if len(weights) == 1 or fw_gap >= away_gap:
            direction = s_vec - x
            alpha_max = 1.0
            is_fw_step = True
        else:
            direction = x - v_vec
            peso_corrente = weights[v_key]

            denom = max(1.0 - peso_corrente, 1e-12)
            alpha_max = peso_corrente / denom
            is_fw_step = False

        # EXACT LINE SEARCH
        if np.max(np.abs(direction)) < 1e-14:
            alpha = 0.0
        else:
            Xd = X @ direction
            den = np.sum(Xd ** 2)
            if den < 1e-10:
                alpha = 0.0
            else:
                alpha_ottimale = -np.dot(grad, direction) / den
                alpha = np.clip(alpha_ottimale, 0.0, alpha_max)

        # UPDATE x
        x = x + alpha * direction

        # UPDATE ACTIVE SET
        if is_fw_step:
            for key in list(weights.keys()):
                weights[key] = (1 - alpha) * weights[key]
            s_key = (s_idx, s_sign)
            weights[s_key] = weights.get(s_key, 0.0) + alpha
        else:
            for key in list(weights.keys()):
                if key == v_key:
                    weights[key] = (1 + alpha) * weights[key] - alpha
                else:
                    weights[key] = (1 + alpha) * weights[key]

        # DROP STEP
        # keys_to_drop = [key for key, val in weights.items() if val < 1e-9]
        # for key in keys_to_drop:
        #     del weights[key]
        drop_tol = 1e-14

        # for key in list(weights):
        #     if weights[key] <= drop_tol:
        #         del weights[key]
        if not is_fw_step and alpha >= alpha_max - drop_tol:
            weights[v_key] = 0.0
            del weights[v_key]

    # NORMALIZZAZIONE FINALE 
    somma = sum(weights.values())
    for key in weights.keys():
        weights[key] /= somma

    cpu_time = time() - t0
    return x, history_loss, history_gap, history_time, history_sparsity

# proviamo a vedere con pairwise

def pairwise_fw_lasso(X, y, tau, max_iters=1000, tolerance=1e-4):
    """
    Pairwise Frank-Wolfe ottimizzato con dizionario per il problema LASSO.
    """
    n_samples, n_features = X.shape
    x = np.zeros(n_features)
    
    # Inizializzazione
    grad_0 = compute_gradient(X, y, x)
    start_idx = np.argmax(np.abs(grad_0))
    start_sign = -np.sign(grad_0[start_idx])
    
    x[start_idx] = start_sign * tau
    weights = {(start_idx, start_sign): 1.0} 

    history_loss = []
    history_gap = []
    history_time = []
    history_sparsity = []
    t0 = time()

    for k in range(max_iters):
        # 1. Normalizzazione di sicurezza
        # somma = sum(weights.values())
        # for key in weights.keys():
        #     weights[key] /= somma

        grad = compute_gradient(X, y, x)
        history_loss.append(compute_loss(X, y, x))

        # --- 2. FW VERTEX (s_t) ---
        s_idx = np.argmax(np.abs(grad))
        s_sign = -np.sign(grad[s_idx])
        s_vec = np.zeros(n_features)
        s_vec[s_idx] = s_sign * tau
        s_key = (s_idx, s_sign)

        # --- 3. AWAY VERTEX (v_t) ---
        max_val = -np.inf
        v_key = None
        for key in weights.keys():
            idx, sign = key
            val = grad[idx] * sign * tau 
            if val > max_val:
                max_val = val
                v_key = key
                
        v_vec = np.zeros(n_features)
        v_idx, v_sign = v_key
        v_vec[v_idx] = v_sign * tau

        # --- 4. STOPPING CONDITION ---
        fw_gap = -np.dot(grad, s_vec - x)
        history_gap.append(fw_gap)
        history_time.append(time() - t0)
        # history_sparsity.append(np.sum(np.abs(x) > 1e-8))
        history_sparsity.append(len(weights))

        if fw_gap <= tolerance:
            print(f"PFW Convergenza raggiunta all'iterazione {k} con gap: {fw_gap:.6f}")
            break

        # --- 5. PAIRWISE DIRECTION ---
        # Trasferimento diretto di massa dal vertice peggiore (v_vec) a quello migliore (s_vec)
        direction = s_vec - v_vec
                
        # La stabilità del PFW: il passo massimo è semplicemente il peso del vertice Away!
        alpha_max = weights[v_key]

        # --- 6. EXACT LINE SEARCH ---
        Xd = X @ direction
        den = np.sum(Xd ** 2)
        if den < 1e-10:
            alpha = 0.0
        else:
            alpha_ottimale = -np.dot(grad, direction) / den
            alpha = np.clip(alpha_ottimale, 0.0, alpha_max)

        # --- 7. UPDATE x ---
        x = x + alpha * direction

        # --- 8. UPDATE ACTIVE SET ---
        # Modifichiamo SOLO i due vertici coinvolti. Tutti gli altri pesi restano invariati.
        weights[v_key] -= alpha
        weights[s_key] = weights.get(s_key, 0.0) + alpha

        # --- 9. DROP STEP ---
        # keys_to_drop = [key for key, val in weights.items() if val < 1e-9]
        # for key in keys_to_drop:
        #     del weights[key]
        drop_tol = 1e-14

        # for key in list(weights):
        #     if weights[key] <= drop_tol:
        #         del weights[key]
        if alpha >= alpha_max - drop_tol:
            weights[v_key] = 0.0
            del weights[v_key]

    cpu_time = time() - t0
    return x, history_loss, history_gap, history_time, history_sparsity


# =============================================
# STEP 3: load the dataset from local JSON file
# =============================================
print("1. Caricamento del dataset Riboflavin...")

file_name = 'data/riboflavin.csv' 
df = pd.read_csv(file_name)

# 2. X AND Y SEPARATION 
target_col = 'y' 

y = df[target_col].values
X = df.drop(columns=[target_col]).values

print(f"Dataset dimensions: {X.shape[0]} observations, {X.shape[1]} features (genes)")

print("\n2. Pre-processing in corso...")
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# no centered_y ma standardizziamo la y (Media 0, Varianza 1)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print("\n3. Esecuzione algoritmo Frank-Wolfe...")
tau_param = 1.0
pesi_fw, loss_fw, gap_fw, time_fw, spars_fw = frank_wolfe_lasso(X_scaled, y_scaled, tau=tau_param)
#tau_param = 50.0 
#pesi_finali, loss_hist, gap_hist = frank_wolfe_lasso(X_scaled, y_scaled, tau=tau_param)

pesi_non_nulli = np.sum(np.abs(pesi_fw) > 1e-8)

print("\n--- RISULTATI FW STANDARD ---")
print(f"Loss finale raggiunta: {loss_fw[-1]:.4f}")
print(f"Gap finale: {gap_fw[-1]:.6f}")
print(f"Feature selezionate (non nulle): {pesi_non_nulli} su {X_scaled.shape[1]}")

print("\n4. Esecuzione algoritmo Away-Step Frank-Wolfe (AFW)...")
pesi_afw, loss_afw, gap_afw, time_afw, spars_afw = away_steps_fw_lasso(X_scaled, y_scaled, tau=tau_param)
#pesi_afw, loss_afw, gap_afw = away_steps_fw_lasso(X_scaled, y_scaled, tau=tau_param)
pesi_afw_non_nulli = np.sum(np.abs(pesi_afw) > 1e-8)

print("\n--- RISULTATI AFW ---")
print(f"Loss iniziale (dopo il 1° vertice): {loss_afw[0]:.4f}")
print(f"Loss finale: {loss_afw[-1]:.4f}")
print(f"Gap finale: {gap_afw[-1]:.6f}")
print(f"Feature selezionate (non nulle): {pesi_afw_non_nulli} su {X_scaled.shape[1]}")

print("\n5. Esecuzione algoritmo Pairwise Frank-Wolfe (PFW)...")
pesi_pfw, loss_pfw, gap_pfw, time_pfw, spars_pfw = pairwise_fw_lasso(X_scaled, y_scaled, tau=tau_param)
pesi_pfw_non_nulli = np.sum(np.abs(pesi_pfw) > 1e-8)

print("\n--- RISULTATI PFW ---")
print(f"Loss iniziale (dopo il 1° vertice): {loss_pfw[0]:.4f}")
print(f"Loss finale: {loss_pfw[-1]:.4f}")
print(f"Gap finale: {gap_pfw[-1]:.6f}")
print(f"Feature selezionate (non nulle): {pesi_pfw_non_nulli} su {X_scaled.shape[1]}")

"""
notiamo che nel pairwise fw, il numero di feature selezionate è maggiore rispetto all'away step fw.
Questo è coerente con la teoria: il pairwise fw può selezionare più vertici perchè permette 
di trasferire peso tra vertici già selezionati.
376 features vuol dire che c'è piu sparsità --> sparsità ed errore di training sono inversamente proporzionali,
infatti abbiamo una loss piu alta rispetto all'away step fw (e lo standard).
Nell'AFW, quando fai un passo, scali percentualmente i pesi di tutti i vertici nell'Active Set. 
Ci vuole molto tempo per far scendere un peso esattamente a zero.
Nel PFW, fai un trasferimento di massa "chirurgico": sottrai peso solo ed esclusivamente al 
vertice peggiore per darlo al vertice migliore.
GEMINI sul gap: Il gap indica quanto siamo lontani dal minimo matematico teorico. Essendo a 122.9, 
il PFW ci sta dicendo: "All'iterazione 1000, non sono ancora arrivato sul fondo della parabola".
Perché? Poiché il PFW limita il suo raggio d'azione spostando massa solo tra due vertici alla volta 
(invece di muoversi liberamente verso l'origine come l'AFW), compie passi direzionali molto piccoli. 
A parità di iterazioni (1000), il PFW è rimasto "indietro" nella minimizzazione pura, perché ha 
speso le sue energie a fare pulizia nell'Active Set. Se lo facessi girare per 3000 iterazioni, 
la loss e il gap crollerebbero allineandosi agli altri.

Nota: con meno variabili selezionate, il modello è più interpretabile e meno soggetto a overfitting, 
ma potrebbe avere una performance leggermente peggiore sul training set.
"""


# ==========================================
# STEP 4: plot
# ==========================================
import matplotlib.pyplot as plt

print("\n6. Generazione dei grafici di convergenza...")

# Creazione di una griglia 2x2

# Colori per variante
color_fw = "#48a1e1"
color_afw = "#ff0ea3"
color_pfw = "#38d238"

# ---------------------------------------------------------
# FIGURA 1: Convergenza della Loss
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(loss_fw, label='Standard FW', color=color_fw, linewidth=2)
plt.plot(loss_afw, label='Away-Step FW', color=color_afw, linewidth=2)
plt.plot(loss_pfw, label='Pairwise FW', color=color_pfw, linewidth=2)

plt.title('Convergenza della Loss')
plt.xlabel('Iterazioni')
plt.ylabel('Loss (Log Scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('image/1_loss_convergence_ribo.png', dpi=300) 

# ---------------------------------------------------------
# FIGURA 2: Duality Gap vs Iterazioni
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(gap_fw, label='Standard FW', color=color_fw, linewidth=2)
plt.plot(gap_afw, label='Away-Step FW', color=color_afw, linewidth=2)
plt.plot(gap_pfw, label='Pairwise FW', color=color_pfw, linewidth=2)

plt.title('Duality Gap vs Iterazioni')
plt.xlabel('Iterazioni')
plt.ylabel('Gap (Log Scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('image/2_duality_gap_ribo.png', dpi=300)

# ---------------------------------------------------------
# FIGURA 3: Sparsità (Feature Attive) vs Iterazioni
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(spars_fw, label='Standard FW', color=color_fw, linewidth=2)
plt.plot(spars_afw, label='Away-Step FW', color=color_afw, linewidth=2)
plt.plot(spars_pfw, label='Pairwise FW', color=color_pfw, linewidth=2)

plt.title('Sparsità (Dimensione Active Set)')
plt.xlabel('Iterazioni')
plt.ylabel('Numero di Feature Non Nulle')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('image/3_sparsity_ribo.png', dpi=300)

# ---------------------------------------------------------
# FIGURA 4: Duality Gap vs CPU Time (Efficienza)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(time_fw, gap_fw, label='Standard FW', color=color_fw, linewidth=2)
plt.plot(time_afw, gap_afw, label='Away-Step FW', color=color_afw, linewidth=2)
plt.plot(time_pfw, gap_pfw, label='Pairwise FW', color=color_pfw, linewidth=2)

plt.title('Efficienza: Gap vs Tempo CPU')
plt.xlabel('Tempo (secondi)')
plt.ylabel('Gap (Log Scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('image/4_cpu_time_ribo.png', dpi=300)

# Mostra tutte le finestre create a schermo contemporaneamente
plt.show()

print("Script completato con successo! I grafici sono stati salvati nella cartella del progetto.")

# altri plot interessanti:
# - plot dei pesi finali (sparse) per vedere quali feature sono state selezionate
# - plot della distribuzione dei pesi (istogramma) per vedere la sparsità e la concentrazione dei pesi
# - plot della loss e del gap in funzione del tempo di esecuzione (per confrontare le velocità di convergenza)

# --- Grafico 5: Distribuzione dei Pesi (Istogramma della Sparsità) ---
print("Generazione dell'istogramma dei pesi...")

# Creiamo una nuova figura per gli istogrammi
fig_hist, axs_hist = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Definiamo una tolleranza numerica per filtrare gli zeri effettivi
tolleranza = 1e-8

# Estraiamo solo i pesi "sopravvissuti" al LASSO
pesi_fw_attivi = pesi_fw[np.abs(pesi_fw) > tolleranza]
pesi_afw_attivi = pesi_afw[np.abs(pesi_afw) > tolleranza]
pesi_pfw_attivi = pesi_pfw[np.abs(pesi_pfw) > tolleranza]

# 1. Istogramma Standard FW
axs_hist[0].hist(pesi_fw_attivi, bins=30, color=color_fw, alpha=0.7, edgecolor='black')
axs_hist[0].set_title(f'Standard FW\n({len(pesi_fw_attivi)} feature attive)')
axs_hist[0].set_xlabel('Valore del Coefficiente')
axs_hist[0].set_ylabel('Frequenza Assoluta')
axs_hist[0].grid(axis='y', linestyle='--', alpha=0.7)

# 2. Istogramma Away-Step FW
axs_hist[1].hist(pesi_afw_attivi, bins=30, color=color_afw, alpha=0.7, edgecolor='black')
axs_hist[1].set_title(f'Away-Step FW\n({len(pesi_afw_attivi)} feature attive)')
axs_hist[1].set_xlabel('Valore del Coefficiente')
axs_hist[1].grid(axis='y', linestyle='--', alpha=0.7)

# 3. Istogramma Pairwise FW
axs_hist[2].hist(pesi_pfw_attivi, bins=30, color=color_pfw, alpha=0.7, edgecolor='black')
axs_hist[2].set_title(f'Pairwise FW\n({len(pesi_pfw_attivi)} feature attive)')
axs_hist[2].set_xlabel('Valore del Coefficiente')
axs_hist[2].grid(axis='y', linestyle='--', alpha=0.7)

fig_hist.suptitle('Distribuzione dei Coefficienti Non Nulli (Magnitudo dei Pesi)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('image/5_weight_distribution_ribo.png', dpi=300)
plt.show()

"""
sull'ultimo grafico (istogramma) vediamo che il lasso porta tutti alla stessa soluzione, perché è un problema 
strettamente convesso e ammette un unico minimo globale.
Il grafico dimostra che, indipendentemente dalla traiettoria scelta e dalla gestione dell'Active Set 
(aggiunta di vertici nel FW standard, rimozione nell'Away-Step o scambio di massa nel Pairwise), 
tutti e tre gli algoritmi sono approdati esattamente alle stesse coordinate spaziali. 
Hanno selezionato le stesse 7 variabili e assegnato loro gli stessi identici pesi (tau = 0.5 ha rimosso
1600 regressori)
NOTA: la distribuzione non è uniforme (uno è infondo a destra), questo nel concreto significa che tra
le pagine Wikipedia selezionate, ne esiste una con un potere predittivo fortemente dominante rispetto alle 
altre nel determinare le visite della pagina target
"""