import numpy as np
import cv2
import matplotlib.pyplot as plt

def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    image = image / 255 if image.max() > 1 else image

    v: np.ndarray = np.max(image, axis=2)
    c: np.ndarray = v - np.min(image, axis=2)
    s: np.ndarray = np.zeros_like(v)

    zero_mask = v > 0
    s[zero_mask] = c[zero_mask] / v[zero_mask]
    r: np.ndarray = image[:, :, 0]
    g: np.ndarray = image[:, :, 1]
    b: np.ndarray = image[:, :, 2]

    h_prime: np.ndarray = np.zeros_like(v)

    nonzero_c_mask = c != 0
    r_mask = (v == r) & nonzero_c_mask
    g_mask = (v == g) & nonzero_c_mask
    b_mask = (v == b) & nonzero_c_mask

    h_prime[r_mask] = ((g[r_mask]  - b[r_mask])/c[r_mask]) % 6
    h_prime[g_mask] = ((b[g_mask] - r[g_mask])/c[g_mask]) + 2
    h_prime[b_mask] = ((r[b_mask] - g[b_mask])/c[b_mask]) + 4

    h: np.ndarray = h_prime * 60
    h[h < 0] += 360

    hsv: np.ndarray = np.stack([h, s, v], axis=2)
    return hsv

