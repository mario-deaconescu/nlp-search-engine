import numpy as np

def l2_normalize(x):
    normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
    return normalized
