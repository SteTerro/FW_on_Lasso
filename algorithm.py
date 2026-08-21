
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
    """
    residual = (X @ x_weights) - y
    return 0.5 * np.sum(residual ** 2)

def compute_gradient(X, y, x_weights):
    """
    Compute the gradient vector of the objective function.
    """
    residual = (X @ x_weights) - y
    return X.T @ residual

# =============================================
# STEP 2: Frank-Wolfe implementation for LASSO
# =============================================

# 1. linear minimization oracle for L1 ball
def lmo_l1(gradient, tau):
    s = np.zeros_like(gradient)
    idx_max = np.argmax(np.abs(gradient))
    s[idx_max] = -np.sign(gradient[idx_max]) * tau
    return s

# 2. Frank-Wolfe algorithm for LASSO
class FrankWolfeLasso:
    def __init__(self, tau, step_size='exact', max_iter=1000, tolerance=1e-4, w_tolerance=1e-8):
        self.tau = tau
        self.step_size = step_size
        self.iter = max_iter
        self.tol = tolerance
        self.w_tolerance = w_tolerance

        self.history_loss = []
        self.history_gap = []
        self.history_time = []
        self.history_sparsity = []
        self.mse = []
        self.history_iter = []

    def update_history(self, gap, start, niter):
        loss = 0.5 * np.sum(self.residual ** 2)
        mse = np.mean(self.residual ** 2)
        self.history_loss.append(loss)
        self.history_gap.append(gap)
        self.history_time.append(time() - start)
        self.history_sparsity.append(np.sum(np.abs(self.x_t) > self.w_tolerance))
        self.mse.append(mse)
        self.history_iter.append(niter)

    def get_history(self):
        return self.history_loss, self.history_gap, self.history_time, self.history_sparsity, self.mse, self.history_iter

    def get_number_non_zero_weights(self):
        return np.sum(np.abs(self.x_t) > self.w_tolerance)

    def get_non_zero_weights(self):
        return self.x_t[np.abs(self.x_t) > self.w_tolerance]

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.x_t

    def mse_score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((np.asarray(y) - y_pred) ** 2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.x_t = np.zeros(n_features) 
        self.residual = -y.copy()

        t0 = time()
    
        for t in range(self.iter):
            # 1. current gradient computation (O(nd))
            grad = X.T @ self.residual

            # 2. vertex selection
            s_idx = np.argmax(np.abs(grad))
            s_sign = -np.sign(grad[s_idx])

            # 4. duality gap
            grad_dot_x = np.dot(grad, self.x_t)
            gap = grad_dot_x - grad[s_idx] * (s_sign * self.tau)
            
            if gap <= self.tol:
                self.update_history(gap, t0, t)
                break
            
            # Efficient X @ d_t computation (O(n))
            X_s = X[:, s_idx] * (s_sign * self.tau)
            Xd = X_s - (self.residual + y)
            
            # 6. line search
            if self.step_size == 'exact':
                den = np.sum(Xd ** 2)
                if den < 1e-10:
                    gamma = 0.0
                else:
                    opt_alpha = gap / den
                    gamma = np.clip(opt_alpha, 0.0, 1.0)
            elif self.step_size == 'diminishing':
                gamma = min(2.0 / (t + 2.0), 1.0)
            elif self.step_size == 'backtracking':
                # Armijo backtracking line search
                beta = 0.5   # shrink factor
                sigma = 0.1  # sufficient decrease parameter
                gamma = 1.0
                current_loss = 0.5 * np.sum(self.residual ** 2)
                dir_dot_grad = -gap  # <grad, d> = <grad, s - x_t> = -gap
                for _ in range(30):
                    new_residual = self.residual + gamma * Xd
                    new_loss = 0.5 * np.sum(new_residual ** 2)
                    if new_loss <= current_loss + sigma * gamma * dir_dot_grad:
                        break
                    gamma *= beta
            else:
                raise ValueError("step_size must be 'exact', 'diminishing', or 'backtracking'")
            
            # Update x_t
            self.x_t *= (1 - gamma)
            self.x_t[s_idx] += gamma * (s_sign * self.tau)

            # Update residual
            self.residual += gamma * Xd

            self.update_history(gap, t0, t)
            
        return self.x_t

# 3. Away-step Frank-Wolfe for LASSO
class AwayStepsFrankWolfeLasso:
    def __init__(self, tau, step_size='exact', max_iter=1000, tolerance=1e-4, w_tolerance=1e-8):
        self.tau = tau
        self.step_size = step_size
        self.iter = max_iter
        self.tol = tolerance
        self.w_tolerance = w_tolerance

        self.history_loss = []
        self.history_gap = []
        self.history_time = []
        self.history_sparsity = []
        self.mse = []
        self.history_iter = []

    def update_history(self, gap, start, niter):
        loss = 0.5 * np.sum(self.residual ** 2)
        mse = np.mean(self.residual ** 2)
        self.history_loss.append(loss)
        self.history_gap.append(gap)
        self.history_time.append(time() - start)
        self.history_sparsity.append(np.sum(np.abs(self.x_t) > self.w_tolerance))
        self.mse.append(mse)
        self.history_iter.append(niter)

    def get_history(self):
        return self.history_loss, self.history_gap, self.history_time, self.history_sparsity, self.mse, self.history_iter

    def get_number_non_zero_weights(self):
        return np.sum(np.abs(self.x_t) > self.w_tolerance)

    def get_non_zero_weights(self):
        return self.x_t[np.abs(self.x_t) > self.w_tolerance]

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.x_t

    def mse_score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((np.asarray(y) - y_pred) ** 2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.x_t = np.zeros(n_features) 
        self.residual = -y.copy()

        # Initialization of active set
        grad_0 = X.T @ self.residual
        start_idx = np.argmax(np.abs(grad_0))
        start_sign = -np.sign(grad_0[start_idx])

        self.x_t[start_idx] = start_sign * self.tau
        self.weights = {(start_idx, start_sign): 1.0}
        
        self.residual += X[:, start_idx] * (start_sign * self.tau)

        t0 = time()
    
        for i in range(self.iter):
            grad = X.T @ self.residual

            # FW VERTEX
            s_idx = np.argmax(np.abs(grad))
            s_sign = -np.sign(grad[s_idx])

            # AWAY VERTEX 
            max_val = -np.inf
            v_key = None
            for key, weight in self.weights.items():
                if weight <= self.w_tolerance:
                    continue
                idx, sign = key
                val = grad[idx] * sign * self.tau
                if val > max_val:
                    max_val = val
                    v_key = key
            
            if v_key is None:
                v_key = list(self.weights.keys())[0]

            v_idx, v_sign = v_key

            # STOPPING CONDITION
            grad_dot_x = np.dot(grad, self.x_t)
            fw_gap = grad_dot_x - grad[s_idx] * (s_sign * self.tau)

            if fw_gap <= self.tol:
                self.update_history(fw_gap, t0, i)   
                break

            away_gap = grad[v_idx] * (v_sign * self.tau) - grad_dot_x

            # Direction and overflow protection
            if len(self.weights) == 1 or fw_gap >= away_gap:
                is_fw_step = True
                alpha_max = 1.0
                X_s = X[:, s_idx] * (s_sign * self.tau)
                Xd = X_s - (self.residual + y)
                dir_dot_grad = -fw_gap
            else:
                is_fw_step = False
                current_weight = self.weights[v_key]
                denom = max(1.0 - current_weight, 1e-12)
                alpha_max = current_weight / denom
                X_v = X[:, v_idx] * (v_sign * self.tau)
                Xd = (self.residual + y) - X_v
                dir_dot_grad = -away_gap

            # LINE SEARCH
            if self.step_size == 'exact':
                den = np.sum(Xd ** 2)
                if den < 1e-10:
                    alpha = 0.0
                else:
                    opt_alpha = -dir_dot_grad / den
                    alpha = np.clip(opt_alpha, 0.0, alpha_max)
            elif self.step_size == 'diminishing':
                gamma = 2.0 / (i + 2.0)
                alpha = min(gamma, alpha_max)
            elif self.step_size == 'backtracking':
                # Armijo backtracking line search
                beta = 0.5
                sigma = 0.1
                alpha = alpha_max
                current_loss = 0.5 * np.sum(self.residual ** 2)
                for _ in range(30):
                    new_residual = self.residual + alpha * Xd
                    new_loss = 0.5 * np.sum(new_residual ** 2)
                    if new_loss <= current_loss + sigma * alpha * dir_dot_grad:
                        break
                    alpha *= beta
                alpha = min(alpha, alpha_max)
            else:
                raise ValueError("step_size must be 'exact', 'diminishing', or 'backtracking'")

            # UPDATE x and residual
            if is_fw_step:
                self.x_t *= (1 - alpha)
                self.x_t[s_idx] += alpha * (s_sign * self.tau)
            else:
                self.x_t *= (1 + alpha)
                self.x_t[v_idx] -= alpha * (v_sign * self.tau)
            
            self.residual += alpha * Xd

            # UPDATE ACTIVE SET
            if is_fw_step:
                for key in list(self.weights.keys()):
                    self.weights[key] = (1 - alpha) * self.weights[key]
                s_key = (s_idx, s_sign)
                self.weights[s_key] = self.weights.get(s_key, 0.0) + alpha
            else:
                for key in list(self.weights.keys()):
                    self.weights[key] = (1 + alpha) * self.weights[key]
                self.weights[v_key] -= alpha

            # DROP STEP — unified tolerance cleanup
            to_drop = [k for k, v in self.weights.items() if v < self.w_tolerance]
            for k in to_drop:
                del self.weights[k]
            
            self.update_history(fw_gap, t0, i)

        return self.x_t

class PairwiseFrankWolfeLasso:
    def __init__(self, tau, step_size = 'exact', max_iter=1000, tolerance=1e-4, w_tolerance=1e-8):
        self.tau = tau
        self.step_size = step_size
        self.iter = max_iter
        self.tol = tolerance
        self.w_tolerance = w_tolerance

        self.history_loss = []
        self.history_gap = []
        self.history_time = []
        self.history_sparsity = []
        self.mse = []
        self.history_iter = []

    def update_history(self, gap, start, niter):
        loss = 0.5 * np.sum(self.residual ** 2)
        mse = np.mean(self.residual ** 2)
        self.history_loss.append(loss)
        self.history_gap.append(gap)
        self.history_time.append(time() - start)
        self.history_sparsity.append(np.sum(np.abs(self.x_t) > self.w_tolerance))
        self.mse.append(mse)
        self.history_iter.append(niter)

    def get_history(self):
        return self.history_loss, self.history_gap, self.history_time, self.history_sparsity, self.mse, self.history_iter

    def get_number_non_zero_weights(self):
        return np.sum(np.abs(self.x_t) > self.w_tolerance)

    def get_non_zero_weights(self):
        return self.x_t[np.abs(self.x_t) > self.w_tolerance]

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.x_t

    def mse_score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((np.asarray(y) - y_pred) ** 2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.x_t = np.zeros(n_features)
        self.residual = -y.copy()
        
        # Initialize the active set with the first vertex
        grad_0 = X.T @ self.residual
        start_idx = np.argmax(np.abs(grad_0))
        start_sign = -np.sign(grad_0[start_idx])
        
        self.x_t[start_idx] = start_sign * self.tau
        self.weights = {(start_idx, start_sign): 1.0} 
        
        self.residual += X[:, start_idx] * (start_sign * self.tau)

        t0 = time()

        for i in range(self.iter):
            grad = X.T @ self.residual

            # 2. FW VERTEX (s_t)
            s_idx = np.argmax(np.abs(grad))
            s_sign = -np.sign(grad[s_idx])
            s_key = (s_idx, s_sign)

            # 3. AWAY VERTEX (v_t)
            max_val = -np.inf
            v_key = None
            for key, weight in self.weights.items():
                if weight <= self.w_tolerance:
                    continue
                idx, sign = key
                val = grad[idx] * sign * self.tau 
                if val > max_val:
                    max_val = val
                    v_key = key
            
            if v_key is None:
                v_key = list(self.weights.keys())[0]
                    
            v_idx, v_sign = v_key

            # 4. STOPPING CONDITION
            grad_dot_x = np.dot(grad, self.x_t)
            fw_gap = grad_dot_x - grad[s_idx] * (s_sign * self.tau)

            if fw_gap <= self.tol:
                self.update_history(fw_gap, t0, i)   
                break

            # 5. PAIRWISE DIRECTION
            X_s = X[:, s_idx] * (s_sign * self.tau)
            X_v = X[:, v_idx] * (v_sign * self.tau)
            Xd = X_s - X_v
            
            dir_dot_grad = grad[s_idx] * (s_sign * self.tau) - grad[v_idx] * (v_sign * self.tau)
                    
            alpha_max = self.weights[v_key]

            # 6. LINE SEARCH
            if self.step_size == 'exact':
                den = np.sum(Xd ** 2)
                if den < 1e-10:
                    alpha = 0.0
                else:
                    opt_alpha = -dir_dot_grad / den
                    alpha = np.clip(opt_alpha, 0.0, alpha_max)
            elif self.step_size == 'diminishing':
                gamma = 2.0 / (i + 2.0)
                alpha = min(gamma, alpha_max)
            elif self.step_size == 'backtracking':
                # Armijo backtracking line search
                beta = 0.5
                sigma = 0.1
                alpha = alpha_max
                current_loss = 0.5 * np.sum(self.residual ** 2)
                for _ in range(30):
                    new_residual = self.residual + alpha * Xd
                    new_loss = 0.5 * np.sum(new_residual ** 2)
                    if new_loss <= current_loss + sigma * alpha * dir_dot_grad:
                        break
                    alpha *= beta
                alpha = min(alpha, alpha_max)
            else:
                raise ValueError("step_size must be 'exact', 'diminishing', or 'backtracking'")

            # 7. UPDATE x and residual
            self.x_t[s_idx] += alpha * (s_sign * self.tau)
            self.x_t[v_idx] -= alpha * (v_sign * self.tau)
            self.residual += alpha * Xd

            # 8. UPDATE ACTIVE SET
            self.weights[v_key] -= alpha
            self.weights[s_key] = self.weights.get(s_key, 0.0) + alpha

            # 9. DROP STEP — unified tolerance cleanup
            to_drop = [k for k, v in self.weights.items() if v < self.w_tolerance]
            for k in to_drop:
                del self.weights[k]

            self.update_history(fw_gap, t0, i)   

        return self.x_t