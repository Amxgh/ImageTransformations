import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from numpy.lib import stride_tricks


def random_square_crop(image: np.ndarray, size:int):
    h, w, c = image.shape

    if size > min(w, h) or size < 0:
        raise ValueError(f"Size is greater than min(w, h) or less than 0. For this image, the dimensions are: {h}x{w}x{c}")

    start_row = random.randint(0, h - size)
    end_row = start_row + size

    start_col = random.randint(0, w - size)
    end_col = start_col + size

    return image[start_row:end_row, start_col:end_col, :]



def main():
    image = cv2.imread("square image.png")
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = random_square_crop(rgb_img, 100)
    bgr_round_trip = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite("Output.jpg", bgr_round_trip)


if __name__ == "__main__":
    main()