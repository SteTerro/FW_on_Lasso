"""
Away-steps Frank-Wolfe (AFW) and Pairwise Frank-Wolfe (PFW)
for the Constrained Lasso problem:

    min_{x}  (1/2) ||Ax - b||^2
    s.t.     ||x||_1 <= tau

The feasible region is the L1-ball of radius tau, whose vertices are:
    A = { +tau * e_i,  -tau * e_i  :  i = 1, ..., n }

where e_i is the i-th canonical basis vector.

References:
    Lacoste-Julien & Jaggi, "On the Global Linear Convergence of
    Frank-Wolfe Optimization Variants", NIPS 2015.
"""

import numpy as np
from time import time


# ================================================================
#  Helper functions
# ================================================================

def compute_loss(X, y, x):
    """Lasso loss: (1/2) ||Xx - y||^2."""
    residual = X @ x - y
    return 0.5 * np.dot(residual, residual)


def compute_gradient(X, y, x):
    """Gradient of (1/2)||Xx - y||^2  =>  X^T (Xx - y)."""
    return X.T @ (X @ x - y)


def lmo_l1_ball(grad, tau):
    """
    Linear Minimisation Oracle over the L1-ball of radius tau.

    Solves:  s = argmin_{||s||_1 <= tau}  <grad, s>

    The solution is  s = -tau * sign(grad_j) * e_j
    where j = argmax_i |grad_i|.

    Returns
    -------
    s       : np.ndarray  — the FW vertex
    v_idx   : int          — index j of the chosen coordinate
    v_sign  : int          — sign (+1 or -1) applied to e_j  (so s = v_sign * tau * e_j)
    """
    j = np.argmax(np.abs(grad))
    sign_j = -np.sign(grad[j]) if grad[j] != 0 else 1.0
    s = np.zeros_like(grad)
    s[j] = sign_j * tau
    return s, j, int(sign_j)


def _vertex_key(idx, sign):
    """Unique hashable key for a vertex ±tau*e_idx."""
    return (idx, sign)


def _exact_line_search_quadratic(X, x, d, y):
    """
    Exact line-search for f(x + gamma * d) = (1/2)||X(x+gamma*d) - y||^2.

    gamma* = - <grad_f(x), d> / ||Xd||^2
           = - <X^T(Xx-y), d> / ||Xd||^2

    The result is clipped to [0, gamma_max] by the caller.
    """
    residual = X @ x - y          # Xx - y
    Xd = X @ d
    numerator = -np.dot(residual, Xd)   # -<Xx-y, Xd> = -<grad, d>  (since grad = X^T(Xx-y))
    denominator = np.dot(Xd, Xd)        # ||Xd||^2
    if denominator < 1e-30:
        return 0.0
    return numerator / denominator


def compute_mse(X, y, x):
    """Mean Squared Error."""
    residual = X @ x - y
    return np.dot(residual, residual) / len(y)


def compute_sparsity(x, tol=1e-8):
    """Number of (near-)zero components."""
    return np.sum(np.abs(x) < tol)


# ================================================================
#  Away-steps Frank-Wolfe  (AFW)
# ================================================================

class AwayStepsFrankWolfeLasso:
    """
    Algorithm 1 — Away-steps Frank-Wolfe for the constrained Lasso.

    Parameters
    ----------
    X        : (m, n) design matrix
    y        : (m,)   target vector
    tau      : float  — L1 radius
    max_iter : int    — maximum number of iterations
    tol      : float  — FW gap tolerance (epsilon)
    """

    def __init__(self, X, y, tau, max_iter=1000, tol=1e-6):
        self.X = X
        self.y = y
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, x0=None):
        """
        Run AFW.

        Parameters
        ----------
        x0 : np.ndarray or None
             Initial feasible point.  If None, uses the origin (which is
             in the L1-ball for any tau > 0; alternatively a vertex is chosen).

        Returns
        -------
        x          : np.ndarray — solution
        history    : dict       — iteration-wise metrics
        """
        X, y, tau = self.X, self.y, self.tau
        m, n = X.shape

        # --- Initialisation (Line 1) ---
        if x0 is not None:
            x = x0.copy()
        else:
            # Start from the LMO solution at x=0 (a vertex of the L1-ball)
            grad0 = compute_gradient(X, y, np.zeros(n))
            x, v0_idx, v0_sign = lmo_l1_ball(grad0, tau)

        # Build the initial active set S^{(0)} and weights alpha
        # x^{(0)} must be expressible as a convex combination of vertices
        active_set = {}   # key -> vertex array
        weights = {}      # key -> alpha_v
        self._decompose_into_vertices(x, active_set, weights)

        # History tracking
        history = {
            'loss': [], 'gap': [], 'time': [], 'iter': [],
            'sparsity': [], 'mse': [], 'step_size': []
        }
        t_start = time()

        for t in range(self.max_iter):
            grad = compute_gradient(X, y, x)

            # --- Line 3: FW direction ---
            s_t, s_idx, s_sign = lmo_l1_ball(grad, tau)
            d_fw = s_t - x                          # d_t^FW

            # --- Line 4: Away direction ---
            v_t_key, v_t = self._away_vertex(grad, active_set)
            d_away = x - v_t                         # d_t^A

            # --- Line 5: FW gap ---
            g_fw = np.dot(-grad, d_fw)               # <-∇f(x), d_FW>
            if g_fw <= self.tol:
                self._record(history, t, x, g_fw, t_start, 0.0)
                break

            # --- Lines 6-10: choose direction ---
            crit_fw = np.dot(-grad, d_fw)
            crit_away = np.dot(-grad, d_away)

            if crit_fw >= crit_away:
                # FW direction (Line 7)
                d_t = d_fw
                gamma_max = 1.0
                is_fw_step = True
                step_vertex_key = _vertex_key(s_idx, s_sign)
                step_vertex = s_t.copy()
            else:
                # Away direction (Line 9)
                d_t = d_away
                alpha_vt = weights[v_t_key]
                gamma_max = alpha_vt / (1.0 - alpha_vt) if alpha_vt < 1.0 else 1e12
                is_fw_step = False
                step_vertex_key = v_t_key
                step_vertex = v_t.copy()

            # --- Line 11: line-search ---
            gamma_opt = _exact_line_search_quadratic(X, x, d_t, y)
            # gamma_opt = 2.0 / (t + 2.0) # min(2.0 / (t + 2.0), gamma_max)
            gamma_t = np.clip(gamma_opt, 0.0, gamma_max)

            # --- Line 12: update x ---
            x = x + gamma_t * d_t

            # --- Update weights (see text) ---
            if is_fw_step:
                # FW step: increase weight of s_t, decrease all others
                for k in weights:
                    weights[k] *= (1.0 - gamma_t)
                sk = _vertex_key(s_idx, s_sign)
                if sk in weights:
                    weights[sk] += gamma_t
                else:
                    weights[sk] = gamma_t
                    active_set[sk] = s_t.copy()
            else:
                # Away step: decrease weight of v_t, increase all others
                for k in weights:
                    weights[k] *= (1.0 + gamma_t)
                weights[v_t_key] -= gamma_t

            # --- Line 13: update active set (remove zero-weight vertices) ---
            to_remove = [k for k, w in weights.items() if w < 1e-12]
            for k in to_remove:
                del weights[k]
                del active_set[k]

            # Normalise weights to avoid numerical drift
            w_sum = sum(weights.values())
            if w_sum > 0:
                for k in weights:
                    weights[k] /= w_sum

            # Record history
            self._record(history, t, x, g_fw, t_start, gamma_t)

        return x, history

    # ------ helpers ------

    def _away_vertex(self, grad, active_set):
        """
        Line 4:  v_t = argmax_{v in S^{(t)}} <∇f(x), v>
        """
        best_key = None
        best_val = -np.inf
        for key, v in active_set.items():
            val = np.dot(grad, v)
            if val > best_val:
                best_val = val
                best_key = key
        return best_key, active_set[best_key]

    def _decompose_into_vertices(self, x, active_set, weights):
        """
        Decompose x in the L1-ball into a convex combination of vertices
        ±tau*e_i.  Each non-zero x_i contributes the vertex sign(x_i)*tau*e_i
        with weight |x_i|/tau.  If x=0 we pick an arbitrary vertex.
        """
        tau = self.tau
        n = len(x)
        active_set.clear()
        weights.clear()

        total = np.sum(np.abs(x))
        if total < 1e-15:
            # x ≈ 0: represent as equal mix (or single vertex)
            v = np.zeros(n)
            v[0] = tau
            key = _vertex_key(0, 1)
            active_set[key] = v
            weights[key] = 1.0
            return

        for i in range(n):
            if np.abs(x[i]) > 1e-15:
                si = 1 if x[i] > 0 else -1
                key = _vertex_key(i, si)
                v = np.zeros(n)
                v[i] = si * tau
                active_set[key] = v
                weights[key] = np.abs(x[i]) / tau

        # If ||x||_1 < tau, we need a "slack" vertex to make weights sum to 1.
        # We add an arbitrary vertex with remaining weight.
        w_sum = sum(weights.values())
        if w_sum < 1.0 - 1e-12:
            # pick a vertex not already in active_set
            for i in range(n):
                for si in [1, -1]:
                    key = _vertex_key(i, si)
                    if key not in active_set:
                        v = np.zeros(n)
                        v[i] = si * tau
                        active_set[key] = v
                        weights[key] = 1.0 - w_sum
                        return
        # normalise
        w_sum = sum(weights.values())
        for k in weights:
            weights[k] /= w_sum

    def _record(self, history, t, x, gap, t_start, gamma):
        history['iter'].append(t)
        history['loss'].append(compute_loss(self.X, self.y, x))
        history['gap'].append(gap)
        history['time'].append(time() - t_start)
        history['sparsity'].append(compute_sparsity(x))
        history['mse'].append(compute_mse(self.X, self.y, x))
        history['step_size'].append(gamma)


# ================================================================
#  Pairwise Frank-Wolfe  (PFW)
# ================================================================

class PairwiseFrankWolfeLasso:
    """
    Algorithm 2 — Pairwise Frank-Wolfe for the constrained Lasso.

    Identical to AFW except lines 6-10 are replaced by:
        d_t = d_t^PFW = s_t - v_t
        gamma_max = alpha_{v_t}

    Parameters
    ----------
    X        : (m, n) design matrix
    y        : (m,)   target vector
    tau      : float  — L1 radius
    max_iter : int    — maximum number of iterations
    tol      : float  — FW gap tolerance (epsilon)
    """

    def __init__(self, X, y, tau, max_iter=1000, tol=1e-6):
        self.X = X
        self.y = y
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, x0=None):
        """
        Run PFW.

        Returns
        -------
        x          : np.ndarray — solution
        history    : dict       — iteration-wise metrics
        """
        X, y, tau = self.X, self.y, self.tau
        m, n = X.shape

        # --- Initialisation (Line 1) ---
        if x0 is not None:
            x = x0.copy()
        else:
            grad0 = compute_gradient(X, y, np.zeros(n))
            x, v0_idx, v0_sign = lmo_l1_ball(grad0, tau)

        # Build initial active set and weights
        active_set = {}
        weights = {}
        self._decompose_into_vertices(x, active_set, weights)

        # History tracking
        history = {
            'loss': [], 'gap': [], 'time': [], 'iter': [],
            'sparsity': [], 'mse': [], 'step_size': []
        }
        t_start = time()

        for t in range(self.max_iter):
            grad = compute_gradient(X, y, x)

            # --- Line 3: FW direction ---
            s_t, s_idx, s_sign = lmo_l1_ball(grad, tau)
            d_fw = s_t - x

            # --- Line 4: Away vertex ---
            v_t_key, v_t = self._away_vertex(grad, active_set)

            # --- Line 5: FW gap ---
            g_fw = np.dot(-grad, d_fw)
            if g_fw <= self.tol:
                self._record(history, t, x, g_fw, t_start, 0.0)
                break

            # --- PFW direction (replaces lines 6-10) ---
            d_t = s_t - v_t                       # d_t^PFW = s_t - v_t
            gamma_max = weights[v_t_key]           # gamma_max = alpha_{v_t}

            # --- Line 11: line-search ---
            gamma_opt = _exact_line_search_quadratic(X, x, d_t, y)
            # gamma_opt = 2.0 / (t + 2.0) # min(2.0 / (t + 2.0), gamma_max)
            gamma_t = np.clip(gamma_opt, 0.0, gamma_max)

            # --- Line 12: update x ---
            x = x + gamma_t * d_t

            # --- Update weights ---
            # PFW: transfer weight from v_t to s_t
            s_key = _vertex_key(s_idx, s_sign)
            weights[v_t_key] -= gamma_t
            if s_key in weights:
                weights[s_key] += gamma_t
            else:
                weights[s_key] = gamma_t
                active_set[s_key] = s_t.copy()

            # --- Line 13: update active set ---
            to_remove = [k for k, w in weights.items() if w < 1e-12]
            for k in to_remove:
                del weights[k]
                del active_set[k]

            # Normalise weights
            w_sum = sum(weights.values())
            if w_sum > 0:
                for k in weights:
                    weights[k] /= w_sum

            # Record
            self._record(history, t, x, g_fw, t_start, gamma_t)

        return x, history

    # ------ helpers (identical to AFW) ------

    def _away_vertex(self, grad, active_set):
        best_key = None
        best_val = -np.inf
        for key, v in active_set.items():
            val = np.dot(grad, v)
            if val > best_val:
                best_val = val
                best_key = key
        return best_key, active_set[best_key]

    def _decompose_into_vertices(self, x, active_set, weights):
        tau = self.tau
        n = len(x)
        active_set.clear()
        weights.clear()

        total = np.sum(np.abs(x))
        if total < 1e-15:
            v = np.zeros(n)
            v[0] = tau
            key = _vertex_key(0, 1)
            active_set[key] = v
            weights[key] = 1.0
            return

        for i in range(n):
            if np.abs(x[i]) > 1e-15:
                si = 1 if x[i] > 0 else -1
                key = _vertex_key(i, si)
                v = np.zeros(n)
                v[i] = si * tau
                active_set[key] = v
                weights[key] = np.abs(x[i]) / tau

        w_sum = sum(weights.values())
        if w_sum < 1.0 - 1e-12:
            for i in range(n):
                for si in [1, -1]:
                    key = _vertex_key(i, si)
                    if key not in active_set:
                        v = np.zeros(n)
                        v[i] = si * tau
                        active_set[key] = v
                        weights[key] = 1.0 - w_sum
                        return
        w_sum = sum(weights.values())
        for k in weights:
            weights[k] /= w_sum

    def _record(self, history, t, x, gap, t_start, gamma):
        history['iter'].append(t)
        history['loss'].append(compute_loss(self.X, self.y, x))
        history['gap'].append(gap)
        history['time'].append(time() - t_start)
        history['sparsity'].append(compute_sparsity(x))
        history['mse'].append(compute_mse(self.X, self.y, x))
        history['step_size'].append(gamma)


# ================================================================
#  Quick test / demo
# ================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data
    m, n = 100, 50
    X = np.random.randn(m, n)
    true_w = np.zeros(n)
    true_w[:5] = [3, -2, 1.5, -1, 0.5]
    y = X @ true_w + 0.1 * np.random.randn(m)

    tau = 5.0
    max_iter = 500
    tol = 1e-8

    print("=" * 60)
    print("  AFW — Away-steps Frank-Wolfe")
    print("=" * 60)
    afw = AwayStepsFrankWolfeLasso(X, y, tau, max_iter=max_iter, tol=tol)
    x_afw, hist_afw = afw.solve()
    print(f"  Final loss:     {hist_afw['loss'][-1]:.6e}")
    print(f"  Final FW gap:   {hist_afw['gap'][-1]:.6e}")
    print(f"  Iterations:     {len(hist_afw['iter'])}")
    print(f"  Sparsity:       {hist_afw['sparsity'][-1]}/{n} zeros")
    print(f"  ||x||_1:        {np.sum(np.abs(x_afw)):.4f}  (tau={tau})")
    print()

    print("=" * 60)
    print("  PFW — Pairwise Frank-Wolfe")
    print("=" * 60)
    pfw = PairwiseFrankWolfeLasso(X, y, tau, max_iter=max_iter, tol=tol)
    x_pfw, hist_pfw = pfw.solve()
    print(f"  Final loss:     {hist_pfw['loss'][-1]:.6e}")
    print(f"  Final FW gap:   {hist_pfw['gap'][-1]:.6e}")
    print(f"  Iterations:     {len(hist_pfw['iter'])}")
    print(f"  Sparsity:       {hist_pfw['sparsity'][-1]}/{n} zeros")
    print(f"  ||x||_1:        {np.sum(np.abs(x_pfw)):.4f}  (tau={tau})")
