import numpy as np
import cv2
import matplotlib.pyplot as plt

def rgb_to_hsv(image: np.ndarray, h_modifier: float, s_modifier: float, v_modifier: float) -> np.ndarray:
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

    hsv: np.ndarray = np.stack([np.clip(h + h_modifier, 0, 360),
                                np.clip(s + s_modifier, 0, 1),
                                np.clip(v + v_modifier, 0 , 1)],
                               axis=2)
    return hsv



def hsv_to_rgb(image: np.ndarray) -> np.ndarray:
    h: np.ndarray = image[:, :, 0]
    s: np.ndarray = image[:, :, 1]
    v: np.ndarray = image[:, :, 2]

    c: np.ndarray = v * s

    h_prime: np.ndarray = h/60

    x: np.ndarray = c * (1 - abs((h_prime % 2) - 1))
    r_prime: np.ndarray = np.zeros_like(h)
    g_prime: np.ndarray = np.zeros_like(h)
    b_prime: np.ndarray = np.zeros_like(h)

    m1 = (0 <= h_prime) & (h_prime < 1) # Mask 1 for condition 1
    r_prime[m1] = c[m1]
    g_prime[m1] = x[m1]
    b_prime[m1] = 0

    m2 = (1 <= h_prime) & (h_prime < 2) # Mask 2 for condition 2
    r_prime[m2] = x[m2]
    g_prime[m2] = c[m2]
    b_prime[m2] = 0

    m3 = (2 <= h_prime) & (h_prime < 3) # Mask 3 for condition 3
    r_prime[m3] = 0
    g_prime[m3] = c[m3]
    b_prime[m3] = x[m3]

    m4 = (3 <= h_prime) & (h_prime < 4) # Mask 4 for condition 4
    r_prime[m4] = 0
    g_prime[m4] = x[m4]
    b_prime[m4] = c[m4]

    m5 = (4 <= h_prime) & (h_prime < 5) # Mask 5 for condition 5
    r_prime[m5] = x[m5]
    g_prime[m5] = 0
    b_prime[m5] = c[m5]

    m6 = (5 <= h_prime) & (h_prime < 6) # Mask 6 for condition 6
    r_prime[m6] = c[m6]
    g_prime[m6] = 0
    b_prime[m6] = x[m6]

    m: np.ndarray = v - c

    rgb_image: np.ndarray = np.stack([r_prime + m, g_prime + m, b_prime + m], axis=2)

    rgb_image = np.clip(rgb_image, 0, 1)

    return rgb_image


def main(argc: int, argv: list):
    if argc != 5:
        raise ValueError("Invalid syntax\n"
                         "python color_space_test.py <filename> "
                         "<hue value modification> <saturation modification> <value modification>")

    try:
        filename: str = argv[1]
        hue: float = float(argv[2])
        saturation: float = float(argv[3])
        value: float = float(argv[4])
    except TypeError:
        raise TypeError("Invalid data type entered. Following are the datatypes:\n"
                        "Filename : String\n"
                        "Hue : Float\n"
                        "Saturation : Float\n"
                        "Value : Float\n")

    if (hue < 0 or hue > 360) or (saturation < 0 or saturation > 1) or (value < 0 or value > 1):
        raise ValueError("Value exceeds bounds. Following are the valid ranges:"
                         "Hue : [0, 360]"
                         "Saturation : [0, 1]"
                         "Value : [0, 1]")

    image = cv2.imread(filename)
    if image is None:
        raise FileNotFoundError(f"Could not load {filename}. Check your path.")

    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hsv_img = rgb_to_hsv(rgb_img, hue, saturation, value)

    rgb_round_trip = hsv_to_rgb(hsv_img)

    rgb_round_trip_uint8 = (rgb_round_trip * 255).astype(np.uint8)
    bgr_round_trip = cv2.cvtColor(rgb_round_trip_uint8, cv2.COLOR_RGB2BGR)

    cv2.imwrite("Output.jpg", bgr_round_trip)


if __name__ == "__main__":
    import sys

    main(len(sys.argv), sys.argv)
