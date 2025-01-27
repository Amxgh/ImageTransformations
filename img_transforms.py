import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from numpy.lib import stride_tricks


def random_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w, c = img.shape

    if size > min(w, h) or size < 0:
        raise ValueError(f"Size is greater than min(w, h) or less than 0. For this image, the dimensions are: {h}x{w}x{c}")

    start_row: int = random.randint(0, h - size)
    end_row: int = start_row + size

    start_col: int = random.randint(0, w - size)
    end_col: int = start_col + size

    return img[start_row:end_row, start_col:end_col, :]

def extract_patch(img: np.ndarray, num_patches: int) -> np.ndarray:
    h, w, c = img.shape
    size: int = h // num_patches
    shape: list = [h // size, w // size] + [size, size, 3]

    # (row, col, patch_row, patch_col)
    strides: list = [size * s for s in img.strides[:2]] + list(img.strides)
    # extract patches
    patches: np.ndarray = stride_tricks.as_strided(img, shape=shape, strides=strides)
    return patches

# def resize_img(img: np.ndarray, factor: int) -> np.ndarray:

def main():
    image = cv2.imread("squareimage.png")
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = random_crop(rgb_img, 100)
    bgr_round_trip = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite("Output.jpg", bgr_round_trip)

    number_of_patches = 100
    patches = extract_patch(rgb_img, number_of_patches)

    rows, cols = patches.shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(patches[i, j].astype(np.uint8))
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()