# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import json
from time import time

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

# test oer vedere se va --> OKK

#print("\n--- TEST PUNTO 2: FUNZIONE OBIETTIVO E GRADIENTE ---")

# Simuliamo un vettore di pesi iniziali (tutti a zero, punto di partenza tipico per FW)
#n_features = X_scaled.shape[1]
#x_initial = np.zeros(n_features)

# Calcoliamo loss e gradiente all'iterazione 0
#loss_0 = compute_loss(X_scaled, y_centered, x_initial)
#grad_0 = compute_gradient(X_scaled, y_centered, x_initial)

#print(f"Loss iniziale (con pesi a zero): {loss_0:.4f}")
#print(f"Forma del vettore gradiente: {grad_0.shape}")

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
    
    history_loss = []
    history_gap = []
    
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
        
        # 5. Criterio di arresto: se il gap è minore della tolleranza, fermati!
        if gap <= tol:
            print(f"Convergenza raggiunta all'iterazione {t} con gap: {gap:.6f}")
            break
            
        # 6. Exact Line Search
        # Trova il passo gamma ottimale minimizzando analiticamente la parabola
        Xd = X @ d_t
        # Aggiungiamo 1e-10 al denominatore per evitare divisioni per zero
        gamma_ottimale = gap / (np.sum(Xd ** 2) + 1e-10) 
        
        # Il passo di FW deve sempre essere compreso tra 0 e 1 (per restare nel dominio)
        gamma = np.clip(gamma_ottimale, 0.0, 1.0)
        
        # 7. Aggiornamento dei coefficienti
        x_t = x_t + gamma * d_t
        
    return x_t, history_loss, history_gap

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

    obj_history = []
    gap_history = []
    t0 = time()

    for k in range(max_iters):
        grad = compute_gradient(X, y, x)
        obj_history.append(compute_loss(X, y, x))

        # FW VERTEX
        s_idx = np.argmax(np.abs(grad))
        s_sign = -np.sign(grad[s_idx])
        s_vec = np.zeros(n_features)
        s_vec[s_idx] = s_sign * tau

        # AWAY VERTEX (La ricerca avviene qui dentro, senza chiamare funzioni esterne!)
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
        gap_history.append(fw_gap)
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


            ## DA CLAUDE:
            # Protezione overflow numerico: quando peso_corrente -> 1, il vero
            # limite matematico di alpha_max = peso_corrente / (1 - peso_corrente)
            # e' +infinito (drop step), NON 1.0. Usare alpha_max = 1.0 in quel
            # caso limiterebbe artificialmente il passo e rallenterebbe la
            # convergenza. Usiamo quindi np.inf, che np.clip gestisce senza
            # problemi, evitando sia il warning di divisione per zero sia il
            # bug di sotto-limitazione del passo.
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

        #Xd = X @ direction
        #den = np.sum(Xd ** 2)
        #if den < 1e-10:
         #   alpha = 0.0
        #else:
         #   alpha_ottimale = -np.dot(grad, direction) / den
          #  alpha = np.clip(alpha_ottimale, 0.0, alpha_max)

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
        keys_to_drop = [key for key, val in weights.items() if val < 1e-9]
        for key in keys_to_drop:
            del weights[key]

    # NORMALIZZAZIONE FINALE (per sicurezza contro errori di arrotondamento)
    somma = sum(weights.values())
    for key in weights.keys():
        weights[key] /= somma

    cpu_time = time() - t0
    return x, obj_history, gap_history

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

    obj_history = []
    gap_history = []
    t0 = time()

    for k in range(max_iters):
        # 1. Normalizzazione di sicurezza
        somma = sum(weights.values())
        for key in weights.keys():
            weights[key] /= somma

        grad = compute_gradient(X, y, x)
        obj_history.append(compute_loss(X, y, x))

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
        gap_history.append(fw_gap)
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
        keys_to_drop = [key for key, val in weights.items() if val < 1e-9]
        for key in keys_to_drop:
            del weights[key]

    cpu_time = time() - t0
    return x, obj_history, gap_history


# =============================================
# STEP 3: load the dataset from local JSON file
# =============================================
print("1. Caricamento del dataset Math Essentials dal file JSON locale...")

file_name = 'wikivital_mathematics.json' 

try:
    with open(file_name, 'r') as file:
        dataset_json = json.load(file)
        
    # Estraiamo i dati ignorando la chiave "edges" (il grafo)
    visite_giornaliere = []
    # Prendiamo i giorni in ordine cronologico (le chiavi numeriche)
    giorni = sorted([k for k in dataset_json.keys() if k.isdigit()], key=int)
    
    for giorno in giorni:
        visite_giornaliere.append(dataset_json[giorno]['y'])
        
    # Creiamo la matrice: 731 giorni (righe) x 1068 pagine (colonne)
    matrice_dati = np.array(visite_giornaliere)
    
    # SETUP PER IL LASSO: decidiamo di predire le visite della 1° pagina (target) 
    # usando le visite di tutte le altre 1067 pagine (feature)
    y_raw = matrice_dati[:, 0]  
    X_raw = matrice_dati[:, 1:] 
    
    print(f"Dimensioni matrice X: {X_raw.shape}")
    print(f"Dimensioni vettore y: {y_raw.shape}")
    
    print("\n2. Pre-processing in corso...")
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    
    # no centered_y ma standardizziamo la y (Media 0, Varianza 1)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
    
    print("\n3. Esecuzione algoritmo Frank-Wolfe...")
    tau_param = 50.0 
    pesi_finali, loss_hist, gap_hist = frank_wolfe_lasso(X_scaled, y_scaled, tau=tau_param)

    pesi_non_nulli = np.sum(np.abs(pesi_finali) > 1e-8)

    print("\n--- RISULTATI FW STANDARD ---")
    print(f"Loss finale raggiunta: {loss_hist[-1]:.4f}")
    print(f"Gap finale: {gap_hist[-1]:.6f}")
    print(f"Feature selezionate (non nulle): {pesi_non_nulli} su {X_scaled.shape[1]}")

except FileNotFoundError:
    print(f"\nERRORE: Non trovo il file '{file_name}'.")
    print("Controlla che il file si chiami esattamente così e sia nella stessa cartella 'ods_project'!")

print("\n4. Esecuzione algoritmo Away-Step Frank-Wolfe (AFW)...")
pesi_afw, loss_afw, gap_afw = away_steps_fw_lasso(X_scaled, y_scaled, tau=tau_param)
pesi_afw_non_nulli = np.sum(np.abs(pesi_afw) > 1e-8)

print("\n--- RISULTATI AFW ---")
print(f"Loss iniziale (dopo il 1° vertice): {loss_afw[0]:.4f}")
print(f"Loss finale: {loss_afw[-1]:.4f}")
print(f"Gap finale: {gap_afw[-1]:.6f}")
print(f"Feature selezionate (non nulle): {pesi_afw_non_nulli} su {X_scaled.shape[1]}")

print("\n5. Esecuzione algoritmo Pairwise Frank-Wolfe (PFW)...")
pesi_pfw, loss_pfw, gap_pfw = pairwise_fw_lasso(X_scaled, y_scaled, tau=tau_param)
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

# Creazione di una figura con due grafici affiancati
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Grafico 1: Convergenza della Funzione Obiettivo ---
ax1.plot(loss_hist, label='Standard FW', color='#1f77b4', linewidth=2)
ax1.plot(loss_afw, label='Away-Step FW', color='#ff7f0e', linewidth=2)
ax1.plot(loss_pfw, label='Pairwise FW', color='#2ca02c', linewidth=2)

ax1.set_title('Convergenza della Funzione Obiettivo (Loss)')
ax1.set_xlabel('Iterazioni')
ax1.set_ylabel('Valore della Loss (f(x))')
ax1.set_yscale('log')  # Usiamo la scala logaritmica sull'asse Y per evidenziare meglio le differenze
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# --- Grafico 2: Frank-Wolfe Duality Gap ---
ax2.plot(gap_hist, label='Standard FW', color="#4d88db", linewidth=2)
ax2.plot(gap_afw, label='Away-Step FW', color="#ff0edf", linewidth=2)
ax2.plot(gap_pfw, label='Pairwise FW', color="#42d932", linewidth=2)

ax2.set_title('Frank-Wolfe Duality Gap (Scala Logaritmica)')
ax2.set_xlabel('Iterazioni')
ax2.set_ylabel('Valore del Gap')
# Usiamo la scala logaritmica sull'asse Y perché il gap scende molto velocemente
# e i dettagli più importanti si trovano vicino allo zero!
ax2.set_yscale('log') 
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# Mostra i grafici a schermo
plt.tight_layout()
plt.show()

print("Script completato con successo!")