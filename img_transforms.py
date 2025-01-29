import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from numpy.lib import stride_tricks
from color_space_test import rgb_to_hsv, hsv_to_rgb
from random import randint, uniform


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

def resize_img(img: np.ndarray, factor: int) -> np.ndarray:
    """
    Resize an image by a given factor.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W, C), where H is the height, W is the width, and C is the number of channels.
    factor : int
        Scaling factor by which to resize the image.

    Returns
    -------
    np.ndarray
        Resized image of shape (new_H, new_W, C), where `new_H = H * factor` and `new_W = W * factor`,
        with the same dtype as the input.
    """
    h, w, c = img.shape

    new_h: int = int(h * factor)
    new_w: int = int(w * factor)

    resized_img: np.ndarray = np.zeros((new_h, new_w, c), dtype=img.dtype)

    scale_x: float = w / new_w
    scale_y: float = h / new_h

    for new_y in range(new_h):
        for new_x in range(new_w):
            original_x: int = min(int(new_x * scale_x), w-1)
            original_y: int = min(int(new_y * scale_y), w-1)

            resized_img[new_y, new_x] = img[original_y, original_x]

    return resized_img

def color_jitter(img: np.ndarray, hue: int, saturation: float, value: float) -> np.ndarray:
    """
    Apply color jitter to an image by modifying its hue, saturation, and value.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W, C), where H is the height, W is the width, and C is the number of channels.
    hue : float
        Hue adjustment value in the range [0, 360].
    saturation : float
        Saturation adjustment factor in the range [0, 1].
    value : float
        Value (brightness) adjustment factor in the range [0, 1].

    Returns
    -------
    np.ndarray
        Color-jittered image of shape (H, W, C) with the same dtype as the input.
    """
    modified_hsv_image: np.ndarray = rgb_to_hsv(img, randint(0, hue), uniform(0, saturation), uniform(0, value))
    modified_rgb_image: np.ndarray = hsv_to_rgb(modified_hsv_image)

    return modified_rgb_image

def main():
    image = cv2.imread("squareimage.png")
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    # image = random_crop(rgb_img, 100)
    # bgr_round_trip = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #
    # cv2.imwrite("Output.jpg", bgr_round_trip)
    #
    # number_of_patches = 5
    # patches = extract_patch(rgb_img, number_of_patches)
    #
    # rows, cols = patches.shape[:2]
    # fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    #
    # for i in range(rows):
    #     for j in range(cols):
    #         axes[i, j].imshow(patches[i, j].astype(np.uint8))
    #         axes[i, j].axis("off")
    #
    # plt.tight_layout()
    # plt.show()
    #
    #
    # resized_image = resize_img(rgb_img, 10)
    # cv2.imwrite("resized_output.jpg", cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

    jittered_image = color_jitter(rgb_img, 360, 1, 1)
    cv2.imwrite("jittered_output.jpg", cv2.cvtColor(jittered_image, cv2.COLOR_BGR2RGB))

    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(patches[i, j].astype(np.uint8))
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()