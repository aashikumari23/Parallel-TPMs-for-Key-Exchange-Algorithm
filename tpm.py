from __future__ import annotations
import numpy as np

class TreeParityMachine:
    def __init__(self, K: int, N: int, L: int, seed: int, rng: np.random.Generator | None = None):
        self.K = K
        self.N = N
        self.L = L

        # Private random weights, independent per party (Section II).
        self._own_rng = rng if rng is not None else np.random.default_rng()
        self.weights = self._own_rng.integers(-L, L + 1, size=(K, N)).astype(np.int64)

        # Shared (public-seeded) input generator -- both parties must be
        # constructed with the same `seed` to remain synchronized.
        self.input_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Output generation (Section II, Eq. 1, Algorithm 2)
    # ------------------------------------------------------------------ #
    def get_output(self, input_arr: np.ndarray):
        """
        Returns (tau, sigma) for the given (or current) input block.
        sigma_i = sign(sum_j w_ij * x_ij); >0 -> 1, <=0 -> -1
        tau = product of all sigma_i
        """
        inp = input_arr
        activation = np.sum(self.weights * inp, axis=1)
        sigma = np.where(activation > 0, 1, -1).astype(np.int64)
        tau = int(np.prod(sigma))
        return tau, sigma

    # ------------------------------------------------------------------ #
    # 1-h random-walk learning rule (Section III-A, Eq. 3, Algorithm 3)
    # ------------------------------------------------------------------ #
    def random_walk_update(self, sigma: np.ndarray, tau_own: int, input_arr: np.ndarray, hop: int):
        """
        Only hidden units whose sigma_i matches the (matched) output tau are
        updated -- this is the standard random-walk rule, generalized to a
        variable step size `hop` instead of the fixed +/-1 step of the
        conventional algorithm.
        """
        mask = (sigma == tau_own)
        if np.any(mask):
            self.weights[mask, :] += hop * input_arr[mask, :]
            np.clip(self.weights, -self.L, self.L, out=self.weights)

    def flat_weight_count(self) -> int:
        return self.K * self.N

    def get_flat_weights(self, index_slice: slice) -> np.ndarray:
        """Return a copy of a slice of the flattened weight vector."""
        return self.weights.flatten()[index_slice].copy()

    def set_flat_weights(self, values: np.ndarray, index_slice: slice):
        """Write corrected weight values back into self.weights for the
        given flat index range."""
        values = np.clip(values, -self.L, self.L)
        flat = self.weights.flatten()
        flat[index_slice] = values
        self.weights = flat.reshape(self.K, self.N)
