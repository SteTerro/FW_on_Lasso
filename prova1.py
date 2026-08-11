# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import json


# STEP 1: implemetare il lasso

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


# STEP 2: implementare il Frank-Wolfe
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


# STEP 3: load the dataset from local JSON file
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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    y_centered = y_raw - np.mean(y_raw)
    
    print("\n3. Esecuzione algoritmo Frank-Wolfe...")
    tau_param = 50.0 
    pesi_finali, loss_hist, gap_hist = frank_wolfe_lasso(X_scaled, y_centered, tau=tau_param)

    pesi_non_nulli = np.sum(np.abs(pesi_finali) > 1e-8)

    print("\n--- RISULTATI FW STANDARD ---")
    print(f"Loss finale raggiunta: {loss_hist[-1]:.4f}")
    print(f"Gap finale: {gap_hist[-1]:.6f}")
    print(f"Feature selezionate (non nulle): {pesi_non_nulli} su {X_scaled.shape[1]}")

except FileNotFoundError:
    print(f"\nERRORE: Non trovo il file '{file_name}'.")
    print("Controlla che il file si chiami esattamente così e sia nella stessa cartella 'ods_project'!")