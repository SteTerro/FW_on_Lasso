import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import json
from time import time

# ============================
# STEP 1: LASSO implementation 
# ============================
def compute_loss(X, y, x_weights):
    """
    Compute the value of the objective function (Least Squares).
    Parameters:
    X : numpy array of shape (n_samples, n_features)
    y : numpy array of shape (n_samples,)
    x_weights : numpy array of shape (n_features,) - current coefficients
    Returns:
    loss : float
    """
    # Residual (X * x) - y
    residual = (X @ x_weights) - y
    
    # 0.5 * L2_norm_squared
    loss = 0.5 * np.sum(residual ** 2)
    # loss = (1/len(y)) * np.sum(residual ** 2)
    return loss

def compute_gradient(X, y, x_weights):
    """
    Compute the gradient vector of the objective function.
    
    Parameters:
    X : numpy array of shape (n_samples, n_features)
    y : numpy array of shape (n_samples,)
    x_weights : numpy array of shape (n_features,) - current coefficients
    
    Returns:
    grad : numpy array of shape (n_features,)
    """
    # (X * x) - y
    residual = (X @ x_weights) - y
    
    # Gradient: X_transposed multiplied by the residual
    grad = X.T @ residual
    return grad

# =============================================
# STEP 2: Frank-Wolfe implementation for LASSO
# =============================================

# 1. linear minimization oracle for L1 ball
def lmo_l1(gradient, tau):
    """
    Linear Minimization Oracle (LMO) for ball L1.
    Find the optimal atom/vertex s_t to minimize the dot product with gradient.
    """
    # s = zero vector of the same shape as gradient
    s = np.zeros_like(gradient)
    
    # find the index of the feature with the maximum absolute gradient
    idx_max = np.argmax(np.abs(gradient))
    
    # assign the value tau to that index with the opposite sign of the gradient 
    # (to point in the direction of maximum descent)
    s[idx_max] = -np.sign(gradient[idx_max]) * tau
    
    return s

# 2. Frank-Wolfe algorithm for LASSO
class FrankWolfeLasso:
    def __init__(self, tau, max_iter=1000, tolerance=1e-4, w_tolerance=1e-8):
        self.tau = tau
        self.iter = max_iter
        self.tol = tolerance
        self.w_tolerance = w_tolerance
        self.convergence = False


        self.history_loss = []
        self.history_gap = []
        self.history_time = []
        self.history_sparsity = []
        self.mse = []

    def update_history(self, X, y, gap, start, mse):
        self.history_loss.append(compute_loss(X, y, self.x_t))
        self.history_gap.append(gap)
        self.history_time.append(time() - start)
        self.history_sparsity.append(np.sum(np.abs(self.x_t) > self.w_tolerance))
        self.mse.append(mse)

    def get_history(self):
        return self.history_loss, self.history_gap, self.history_time, self.history_sparsity, self.mse

    def get_number_non_zero_weights(self):
        return np.sum(np.abs(self.x_t) > self.w_tolerance)

    def get_non_zero_weights(self):
        return self.x_t[np.abs(self.x_t) > self.w_tolerance]

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.x_t

    def predict_original_scale(self, X, y_scaler):
        y_pred_scaled = self.predict(X)

        return y_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()

    def mse_score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((np.asarray(y) - y_pred) ** 2)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # we start wiht the center of the L1 ball is the zero vector, 
        # which is also the point of maximum sparsity
        self.x_t = np.zeros(n_features) 

        t0 = time()
    
        for t in range(self.iter):
            # 1. current gradient computation
            grad = compute_gradient(X, y, self.x_t)

            # 2. vertex selection, through oracle call
            s_t = lmo_l1(grad, self.tau)

            # 3.ipdate direction
            d_t = s_t - self.x_t

            # 4. duality gap (for stopping condition)
            gap = np.dot(grad, -d_t)
            
            # 5. stopping criterion: if the gap is smaller than the tolerance, we stop
            if gap <= self.tol:
                print(f"self.convergence obtained at iteration {t} with gap: {gap:.6f}")
                self.convergence = True
            
            # 6. line search, choose gamma (learning rate)
            Xd = X @ d_t
            opt_gamma = gap / (np.sum(Xd ** 2) + 1e-10) 
            
            gamma = np.clip(opt_gamma, 0.0, 1.0)
            
            self.x_t = self.x_t + gamma * d_t

            mse = self.mse_score(X, y)

            self.update_history(X, y, gap, t0, mse)

            if self.convergence:
                break
            
        return self.x_t

# 3. Away-step Frank-Wolfe for LASSO
class AwayStepsFrankWolfeLasso:
    def __init__(self, tau, max_iter=1000, tolerance=1e-4, w_tolerance=1e-8):
        self.tau = tau
        self.iter = max_iter
        self.tol = tolerance
        self.w_tolerance = w_tolerance
        self.convergence = False

        self.history_loss = []
        self.history_gap = []
        self.history_time = []
        self.history_sparsity = []
        self.mse = []

    def update_history(self, X, y, gap, start, mse):
        self.history_loss.append(compute_loss(X, y, self.x_t))
        self.history_gap.append(gap)
        self.history_time.append(time() - start)
        self.history_sparsity.append(np.sum(np.abs(self.x_t) > self.w_tolerance))
        self.mse.append(mse)

    def get_history(self):
        return self.history_loss, self.history_gap, self.history_time, self.history_sparsity, self.mse

    def get_number_non_zero_weights(self):
        return np.sum(np.abs(self.x_t) > self.w_tolerance)

    def get_non_zero_weights(self):
        return self.x_t[np.abs(self.x_t) > self.w_tolerance]

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.x_t

    def predict_original_scale(self, X, y_scaler):
        y_pred_scaled = self.predict(X)

        return y_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()

    def mse_score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((np.asarray(y) - y_pred) ** 2)

    def exact_line_search(self, X, y, grad, direction, alpha_max):
        if np.max(np.abs(direction)) < 1e-14:
            alpha = 0.0
        else:
            Xd = X @ direction
            den = np.sum(Xd ** 2)
            if den < 1e-10:
                alpha = 0.0
            else:
                opt_alpha = -np.dot(grad, direction) / den
                alpha = np.clip(opt_alpha, 0.0, alpha_max)
        return alpha

    def line_search(self, X, y, grad, direction, alpha_max):
        Xd = X @ direction
        den = np.sum(Xd ** 2)
        if den < 1e-10:
            alpha = 0.0
        else:
            opt_alpha = -np.dot(grad, direction) / den
            alpha = np.clip(opt_alpha, 0.0, alpha_max)
        return alpha
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.x_t = np.zeros(n_features) 

        # initialization of active set
        grad_0 = compute_gradient(X, y, self.x_t)
        start_idx = np.argmax(np.abs(grad_0))
        start_sign = -np.sign(grad_0[start_idx])

        self.x_t[start_idx] = start_sign * self.tau

        # Active Set (dict: key = index, value = sign)
        self.weights = {(start_idx, start_sign): 1.0}

        t0 = time()
    
        for i in range(self.iter):
            grad = compute_gradient(X, y, self.x_t)

            # FW VERTEX
            s_idx = np.argmax(np.abs(grad))
            s_sign = -np.sign(grad[s_idx])
            s_vec = np.zeros(n_features)
            s_vec[s_idx] = s_sign * self.tau

            # AWAY VERTEX 
            max_val = -np.inf
            v_key = None
            for key in self.weights.keys():
                idx, sign = key
                val = grad[idx] * sign * self.tau
                if val > max_val:
                    max_val = val
                    v_key = key

            v_vec = np.zeros(n_features)
            v_idx, v_sign = v_key
            v_vec[v_idx] = v_sign * self.tau

            # STOPPING CONDITION
            fw_gap = -np.dot(grad, s_vec - self.x_t)

            if fw_gap <= self.tol:
                print(f"self.convergence obtained at iteration {i} with gap: {fw_gap:.6f}")
                self.convergence = True

            # direction and overflow protection
            away_gap = -np.dot(grad, self.x_t - v_vec)

            # protection: if there is only one vertex, or if the FW gap is larger, take FW step
            if len(self.weights) == 1 or fw_gap >= away_gap:
                direction = s_vec - self.x_t
                alpha_max = 1.0
                is_fw_step = True
            else:
                direction = self.x_t - v_vec
                current_weight = self.weights[v_key]

                denom = max(1.0 - current_weight, 1e-12)
                alpha_max = current_weight / denom
                is_fw_step = False

            # EXACT LINE SEARCH
            alpha = self.line_search(X, y, grad, direction, alpha_max)

            # UPDATE x
            self.x_t = self.x_t + alpha * direction

            # UPDATE ACTIVE SET
            if is_fw_step:
                for key in list(self.weights.keys()):
                    self.weights[key] = (1 - alpha) * self.weights[key]
                s_key = (s_idx, s_sign)
                self.weights[s_key] = self.weights.get(s_key, 0.0) + alpha
            else:
                for key in list(self.weights.keys()):
                    if key == v_key:
                        self.weights[key] = (1 + alpha) * self.weights[key] - alpha
                    else:
                        self.weights[key] = (1 + alpha) * self.weights[key]

            # DROP STEP
            to_drop = []
            for key in self.weights:
                if abs(self.weights[key]) < 1e-10:
                    to_drop.append(key)
            for key in to_drop:
                del self.weights[key]

            # keys_to_drop = [key for key, val in weights.items() if val < 1e-9]
            # for key in keys_to_drop:
            #     del weights[key]
            # drop_tol = 1e-14

            # for key in list(weights):
            #     if weights[key] <= drop_tol:
            #         del weights[key]
            # if not is_fw_step and alpha >= alpha_max - drop_tol:
            #     self.weights[v_key] = 0.0
            #     del self.weights[v_key]

            mse = self.mse_score(X, y)  

            self.update_history(X, y, fw_gap, t0, mse)   

            if self.convergence:
                break

        return self.x_t

class PairwiseFrankWolfeLasso:
    def __init__(self, tau, max_iter=1000, tolerance=1e-4, w_tolerance=1e-8):
        self.tau = tau
        self.iter = max_iter
        self.tol = tolerance
        self.w_tolerance = w_tolerance
        self.convergence = False


        self.history_loss = []
        self.history_gap = []
        self.history_time = []
        self.history_sparsity = []
        self.mse = []

    def update_history(self, X, y, gap, start, mse):
        self.history_loss.append(compute_loss(X, y, self.x_t))
        self.history_gap.append(gap)
        self.history_time.append(time() - start)
        self.history_sparsity.append(np.sum(np.abs(self.x_t) > self.w_tolerance))
        self.mse.append(mse)

    def get_history(self):
        return self.history_loss, self.history_gap, self.history_time, self.history_sparsity, self.mse

    def get_number_non_zero_weights(self):
        return np.sum(np.abs(self.x_t) > self.w_tolerance)

    def get_non_zero_weights(self):
        return self.x_t[np.abs(self.x_t) > self.w_tolerance]

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.x_t

    def predict_original_scale(self, X, y_scaler):
        y_pred_scaled = self.predict(X)

        return y_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()

    def mse_score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((np.asarray(y) - y_pred) ** 2)

    def line_search(self, X, y, grad, direction, alpha_max):
        Xd = X @ direction
        den = np.sum(Xd ** 2)
        if den < 1e-10:
            alpha = 0.0
        else:
            opt_alpha = -np.dot(grad, direction) / den
            alpha = np.clip(opt_alpha, 0.0, alpha_max)
        return alpha
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.x_t = np.zeros(n_features)
        
        # Initialize the active set with the first vertex
        grad_0 = compute_gradient(X, y, self.x_t)
        start_idx = np.argmax(np.abs(grad_0))
        start_sign = -np.sign(grad_0[start_idx])
        
        self.x_t[start_idx] = start_sign * self.tau
        self.weights = {(start_idx, start_sign): 1.0} 

        t0 = time()

        for i in range(self.iter):
            grad = compute_gradient(X, y, self.x_t)

            # 2. FW VERTEX (s_t)
            s_idx = np.argmax(np.abs(grad))
            s_sign = -np.sign(grad[s_idx])
            s_vec = np.zeros(n_features)
            s_vec[s_idx] = s_sign * self.tau
            s_key = (s_idx, s_sign)

            # 3. AWAY VERTEX (v_t)
            max_val = -np.inf
            v_key = None
            for key in self.weights.keys():
                idx, sign = key
                val = grad[idx] * sign * self.tau 
                if val > max_val:
                    max_val = val
                    v_key = key
                    
            v_vec = np.zeros(n_features)
            v_idx, v_sign = v_key
            v_vec[v_idx] = v_sign * self.tau

            # 4. STOPPING CONDITION
            fw_gap = -np.dot(grad, s_vec - self.x_t)

            if fw_gap <= self.tol:
                print(f"PFW self.convergence obtained at iteration {i} with gap: {fw_gap:.6f}")
                self.convergence = True

            # 5. PAIRWISE DIRECTION
            # direct transfer of mass from the worst vertex (v_vec) to the best vertex (s_vec)
            direction = s_vec - v_vec
                    
            # the stability of the PFW: the maximum step is simply the weight of the Away vertex!
            alpha_max = self.weights[v_key]

            # 6. EXACT LINE SEARCH
            alpha = self.line_search(X, y, grad, direction, alpha_max)

            # 7. UPDATE x
            self.x_t = self.x_t + alpha * direction

            # 8. UPDATE ACTIVE SET
            # We update the weights of the active set. The weight of the Away vertex decreases,
            # while the weight of the FW vertex increases.
            self.weights[v_key] -= alpha
            self.weights[s_key] = self.weights.get(s_key, 0.0) + alpha

            # 9. DROP STEP
            # keys_to_drop = [key for key, val in weights.items() if val < 1e-9]
            # for key in keys_to_drop:
            #     del weights[key]
            drop_tol = 1e-14

            # for key in list(weights):
            #     if weights[key] <= drop_tol:
            #         del weights[key]
            if alpha >= alpha_max - drop_tol:
                self.weights[v_key] = 0.0
                del self.weights[v_key]

            mse = self.mse_score(X, y) 

            self.update_history(X, y, fw_gap, t0, mse)  

            if self.convergence:
                break          

        return self.x_t

