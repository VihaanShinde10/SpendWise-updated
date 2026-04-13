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
        return np.array([
            self.token_count, self.char_length, self.has_url_flag,
            self.log_merchant_freq, self.semantic_confidence,
            self.recurrence_strength, self.is_new_user
        ], dtype=np.float32)


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
        h = np.maximum(0, self.W1 @ x + self.b1)   # ReLU
        out = self.W2 @ h + self.b2
        alpha = 1 / (1 + np.exp(-out[0]))           # Sigmoid
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

        for epoch in range(epochs):
            idx = np.random.permutation(len(X_tr))
            epoch_loss = []
            for start in range(0, len(X_tr), batch_size):
                batch_idx = idx[start:start+batch_size]
                xb, yb = X_tr[batch_idx], y_tr[batch_idx]

                # Forward
                H = np.maximum(0, (self.W1 @ xb.T).T + self.b1)  # (B, 16)
                pred = (self.W2 @ H.T + self.b2).T.squeeze()       # (B,)
                if pred.ndim == 0:
                    pred = pred.reshape(1)
                alpha_pred = 1 / (1 + np.exp(-pred))

                # MSE loss
                diff = alpha_pred - yb
                loss = float(np.mean(diff**2))
                epoch_loss.append(loss)

                # Backward
                t += 1
                d_out = 2 * diff / len(xb) * alpha_pred * (1 - alpha_pred)
                dW2 = d_out.reshape(1, -1) @ H
                db2 = np.array([np.sum(d_out)])
                dH = d_out.reshape(-1, 1) * self.W2
                dH_relu = dH * (H > 0).astype(float)
                dW1 = dH_relu.T @ xb
                db1 = dH_relu.sum(axis=0)

                def adam_update(param, grad, m, v):
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * grad**2
                    m_hat = m / (1 - beta1**t)
                    v_hat = v / (1 - beta2**t)
                    return param - lr * m_hat / (np.sqrt(v_hat) + eps), m, v

                self.W1, m_W1, v_W1 = adam_update(self.W1, dW1, m_W1, v_W1)
                self.b1, m_b1, v_b1 = adam_update(self.b1, db1, m_b1, v_b1)
                self.W2, m_W2, v_W2 = adam_update(self.W2, dW2, m_W2, v_W2)
                self.b2, m_b2, v_b2 = adam_update(self.b2, db2, m_b2, v_b2)

            history['train_mse'].append(float(np.mean(epoch_loss)))

            # Validation MSE
            if len(X_val) > 0:
                H_val = np.maximum(0, (self.W1 @ X_val.T).T + self.b1)
                pred_val = (self.W2 @ H_val.T + self.b2).T.squeeze()
                if pred_val.ndim == 0:
                    pred_val = pred_val.reshape(1)
                a_val = 1 / (1 + np.exp(-pred_val))
                val_mse = float(np.mean((a_val - y_val)**2))
                history['val_mse'].append(val_mse)

        return history

    def fuse(self, z_sem: np.ndarray, z_beh: np.ndarray, alpha: float) -> np.ndarray:
        """Equation 5: z_final = alpha * z_sem + (1 - alpha) * z_beh"""
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
        net = cls()
        net.W1 = np.array(d['W1'], dtype=np.float32)
        net.b1 = np.array(d['b1'], dtype=np.float32)
        net.W2 = np.array(d['W2'], dtype=np.float32)
        net.b2 = np.array([d['b2']], dtype=np.float32)
        return net
