import numpy as np

class HistoryBuffer:
    def __init__(self, capacity: int, feature_dim: int, batch_size: int = 1):
        if capacity <= 0 or feature_dim <= 0 or batch_size <= 0:
            raise ValueError("capacity, feature_dim and batch_size must be positive")
        self.capacity = capacity
        self.feature_dim = feature_dim
        self._data = np.zeros((batch_size, capacity, feature_dim), dtype=np.float32)
        self._timestamps = np.zeros((batch_size, capacity), dtype=np.int64)
        self._initialized = False

    def reset(self, initial_frame: np.ndarray, timestamp_ns: int):
        frame = self._validate_frame(initial_frame)
        self._data[...] = frame[:, None, :]
        self._timestamps[...] = int(timestamp_ns)
        self._initialized = True

    def push(self, frame: np.ndarray, timestamp_ns: int) -> np.ndarray:
        frame = self._validate_frame(frame)
        if not self._initialized:
            self.reset(frame, timestamp_ns)
            return self.view()
        self._data[:, :-1, :] = self._data[:, 1:, :]
        self._data[:, -1, :] = frame
        self._timestamps[:, :-1] = self._timestamps[:, 1:]
        self._timestamps[:, -1] = int(timestamp_ns)
        return self.view()

    def latest(self) -> np.ndarray:
        return self._data[:, -1, :].copy()

    def view(self) -> np.ndarray:
        return self._data.copy()

    def _validate_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (self._data.shape[0], self.feature_dim):
            raise ValueError(f"expected frame {(self._data.shape[0], self.feature_dim)}, got {frame.shape}")
        if not np.isfinite(frame).all():
            raise ValueError("history frame must be finite")
        return frame

def validate_timestamp_pair(estimator_ts: int, policy_ts: int, tolerance_ns: int = 0) -> None:
    if abs(int(estimator_ts) - int(policy_ts)) > tolerance_ns:
        raise ValueError(f"estimator/policy timestamp skew exceeds tolerance: {estimator_ts} vs {policy_ts}")
