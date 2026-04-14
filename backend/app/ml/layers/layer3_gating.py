"""
Layer 3: Two-layer MLP adaptive gating network (145 parameters).
Implements Section 4.6 of the paper (Equations 4, 5).
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class QualityIndicator:
    """7-dimensional input vector x to the gating network."""
    token_count: float
    char_length: float
    has_url_flag: float         # binary
    log_merchant_freq: float
    semantic_confidence: float  # C_sem
    recurrence_strength: float  # S_rec
    is_new_user: float          # binary (< 15 lifetime txns)

    def to_vector(self) -> np.ndarray:
        vec = np.array([
            self.token_count, self.char_length, self.has_url_flag,
            self.log_merchant_freq, self.semantic_confidence,
            self.recurrence_strength, self.is_new_user
        ], dtype=np.float32)
        if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
            raise ValueError(
                f"QualityIndicator contains NaN or Inf values: {vec}"
            )
        return vec


class GatingNetwork:
    """
    Two-layer MLP: 7 → 16 → 1 (sigmoid output)
    Total parameters: (7×16)+16 + (16×1)+1 = 145
    """

    def __init__(self):
        # He initialisation for ReLU networks
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((16, 7)).astype(np.float32) * np.sqrt(2 / 7)
        self.b1 = np.zeros(16, dtype=np.float32)
        self.W2 = rng.standard_normal((1, 16)).astype(np.float32) * np.sqrt(2 / 16)
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, x: np.ndarray) -> float:
        """Equation 4: alpha = sigmoid(W2 * ReLU(W1 * x + b1) + b2)"""
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (7,):
            raise ValueError(
                f"Expected input shape (7,), got {x.shape}"
            )
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            raise ValueError("Input vector contains NaN or Inf values.")

        h = np.maximum(0, self.W1 @ x + self.b1)       # ReLU
        out = self.W2 @ h + self.b2
        # Clip logit for numerical stability before sigmoid
        out_clipped = np.clip(out[0], -500.0, 500.0)
        alpha = 1.0 / (1.0 + np.exp(-out_clipped))     # Sigmoid
        return float(alpha)

    def generate_pseudolabels(
        self, semantic_conf: float, recurrence_strength: float
    ) -> Optional[float]:
        """
        Self-supervised target generation:
        C_sem > 0.90 → target = 0.90 (semantic dominant)
        S_rec > 0.90 → target = 0.10 (behavioural dominant)
        else         → target = 0.50 (balanced)
        """
        semantic_conf = float(semantic_conf)
        recurrence_strength = float(recurrence_strength)

        if not (0.0 <= semantic_conf <= 1.0):
            raise ValueError(
                f"semantic_conf must be in [0, 1], got {semantic_conf}"
            )
        if not (0.0 <= recurrence_strength <= 1.0):
            raise ValueError(
                f"recurrence_strength must be in [0, 1], got {recurrence_strength}"
            )

        if semantic_conf > 0.90:
            return 0.90
        elif recurrence_strength > 0.90:
            return 0.10
        else:
            return 0.50

    def train(
        self,
        X: np.ndarray,  # shape: (N, 7)
        targets: np.ndarray,  # shape: (N,)
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 64
    ) -> dict:
        """Adam optimiser training (Section 4.6.2)."""
        X = np.asarray(X, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)

        if X.ndim != 2 or X.shape[1] != 7:
            raise ValueError(
                f"X must have shape (N, 7), got {X.shape}"
            )
        if targets.ndim != 1 or len(targets) != len(X):
            raise ValueError(
                f"targets must have shape (N,) matching X rows, "
                f"got targets shape {targets.shape} vs X rows {len(X)}"
            )
        if len(X) == 0:
            raise ValueError("Training data X is empty.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Training data X contains NaN or Inf values.")
        if np.any(np.isnan(targets)) or np.any(np.isinf(targets)):
            raise ValueError("Targets contain NaN or Inf values.")

        # Adam state
        m_W1 = np.zeros_like(self.W1); v_W1 = np.zeros_like(self.W1)
        m_b1 = np.zeros_like(self.b1); v_b1 = np.zeros_like(self.b1)
        m_W2 = np.zeros_like(self.W2); v_W2 = np.zeros_like(self.W2)
        m_b2 = np.zeros_like(self.b2); v_b2 = np.zeros_like(self.b2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        n = len(X)
        val_split = max(1, int(0.8 * n))
        X_tr, X_val = X[:val_split], X[val_split:]
        y_tr, y_val = targets[:val_split], targets[val_split:]
        history = {'train_mse': [], 'val_mse': []}

        # Early stopping state
        best_val_mse = float('inf')
        patience = 20
        patience_counter = 0
        best_W1, best_b1, best_W2, best_b2 = (
            self.W1.copy(), self.b1.copy(),
            self.W2.copy(), self.b2.copy()
        )

        for epoch in range(epochs):
            idx = np.random.permutation(len(X_tr))
            epoch_loss = []

            for start in range(0, len(X_tr), batch_size):
                batch_idx = idx[start:start + batch_size]
                xb, yb = X_tr[batch_idx], y_tr[batch_idx]

                # Forward
                H = np.maximum(0, (self.W1 @ xb.T).T + self.b1)  # (B, 16)
                pred = ((self.W2 @ H.T) + self.b2).T              # (B, 1)
                pred = pred.reshape(-1)                            # (B,)
                alpha_pred = 1.0 / (1.0 + np.exp(-np.clip(pred, -500.0, 500.0)))

                # MSE loss
                diff = alpha_pred - yb
                loss = float(np.mean(diff ** 2))
                epoch_loss.append(loss)

                # Backward
                t += 1
                d_out = (2.0 * diff / len(xb)) * alpha_pred * (1.0 - alpha_pred)
                dW2 = d_out.reshape(1, -1) @ H            # (1, 16)
                db2 = np.array([np.sum(d_out)])            # (1,)
                dH = d_out.reshape(-1, 1) * self.W2       # (B, 16)
                dH_relu = dH * (H > 0).astype(np.float32) # (B, 16)
                dW1 = dH_relu.T @ xb                      # (16, 7)
                db1 = dH_relu.sum(axis=0)                 # (16,)

                # Gradient clipping (max norm = 5.0)
                grads = [dW1, db1, dW2, db2]
                total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
                clip_coef = 5.0 / (total_norm + 1e-8)
                if clip_coef < 1.0:
                    dW1, db1, dW2, db2 = [g * clip_coef for g in grads]

                # Adam update — t passed explicitly to avoid closure bug
                def adam_update(param, grad, m, v, t_step):
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * grad ** 2
                    m_hat = m / (1 - beta1 ** t_step)
                    v_hat = v / (1 - beta2 ** t_step)
                    return param - lr * m_hat / (np.sqrt(v_hat) + eps), m, v

                self.W1, m_W1, v_W1 = adam_update(self.W1, dW1, m_W1, v_W1, t)
                self.b1, m_b1, v_b1 = adam_update(self.b1, db1, m_b1, v_b1, t)
                self.W2, m_W2, v_W2 = adam_update(self.W2, dW2, m_W2, v_W2, t)
                self.b2, m_b2, v_b2 = adam_update(self.b2, db2, m_b2, v_b2, t)

            history['train_mse'].append(float(np.mean(epoch_loss)))

            # Validation MSE + early stopping
            if len(X_val) > 0:
                H_val = np.maximum(0, (self.W1 @ X_val.T).T + self.b1)
                pred_val = ((self.W2 @ H_val.T) + self.b2).T.reshape(-1)
                a_val = 1.0 / (1.0 + np.exp(-np.clip(pred_val, -500.0, 500.0)))
                val_mse = float(np.mean((a_val - y_val) ** 2))
                history['val_mse'].append(val_mse)

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    patience_counter = 0
                    best_W1, best_b1 = self.W1.copy(), self.b1.copy()
                    best_W2, best_b2 = self.W2.copy(), self.b2.copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        # Restore best weights and stop
                        self.W1, self.b1 = best_W1, best_b1
                        self.W2, self.b2 = best_W2, best_b2
                        break

        return history

    def fuse(self, z_sem: np.ndarray, z_beh: np.ndarray, alpha: float) -> np.ndarray:
        """Equation 5: z_final = alpha * z_sem + (1 - alpha) * z_beh"""
        alpha = float(alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"alpha must be in [0, 1], got {alpha}"
            )
        z_sem = np.asarray(z_sem, dtype=np.float32)
        z_beh = np.asarray(z_beh, dtype=np.float32)
        if z_sem.shape != z_beh.shape:
            raise ValueError(
                f"z_sem and z_beh must have the same shape, "
                f"got {z_sem.shape} and {z_beh.shape}"
            )
        return alpha * z_sem + (1 - alpha) * z_beh

    def to_dict(self) -> dict:
        return {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': float(self.b2[0])
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GatingNetwork':
        required_keys = {'W1', 'b1', 'W2', 'b2'}
        missing = required_keys - d.keys()
        if missing:
            raise KeyError(
                f"Missing keys in dict for GatingNetwork: {missing}"
            )
        net = cls()
        net.W1 = np.array(d['W1'], dtype=np.float32)
        net.b1 = np.array(d['b1'], dtype=np.float32)
        net.W2 = np.array(d['W2'], dtype=np.float32)
        net.b2 = np.array([d['b2']], dtype=np.float32)

        if net.W1.shape != (16, 7):
            raise ValueError(
                f"W1 shape mismatch: expected (16, 7), got {net.W1.shape}"
            )
        if net.b1.shape != (16,):
            raise ValueError(
                f"b1 shape mismatch: expected (16,), got {net.b1.shape}"
            )
        if net.W2.shape != (1, 16):
            raise ValueError(
                f"W2 shape mismatch: expected (1, 16), got {net.W2.shape}"
            )
        if net.b2.shape != (1,):
            raise ValueError(
                f"b2 shape mismatch: expected (1,), got {net.b2.shape}"
            )
        return net