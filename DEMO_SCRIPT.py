import cv2
from color_space_test import rgb_to_hsv, hsv_to_rgb
from img_transforms import random_crop, extract_patch, resize_img, color_jitter
from create_img_pyramid import create_img_pyramid
import matplotlib.pyplot as plt
import numpy as np
import os


def demo_rgb_hsv_rgb_conversion(img: np.ndarray, h_modifier: int = 0, s_modifier: float = 0,
                                v_modifier: float = 0) -> None:
    hsv_img = rgb_to_hsv(img, h_modifier, s_modifier, v_modifier)
    if not os.path.exists("demo_output/rgb-hsv-converter"):
        os.makedirs("demo_output/rgb-hsv-converter")

    rgb_roundtrip_img = hsv_to_rgb(hsv_img)

    fig, axes = plt.subplots(1, 3)

    axes[0].imshow(img)
    axes[0].set_title("Original RGB Image")
    axes[0].axis(False)

    axes[1].imshow(hsv_img)
    axes[1].set_title("HSV Image")
    axes[1].axis(False)

    axes[2].imshow(rgb_roundtrip_img)
    axes[2].set_title("Round-trip RGB Image")
    axes[2].axis(False)

    plt.suptitle("RGB to HSV to RGB", fontsize=16)
    modifiers = "No HSV Modifiers" if not any((h_modifier, s_modifier, v_modifier)) else (
        f"Hue Modifier: {h_modifier}\n"
        f"Saturation Modifier: {s_modifier}\n"
        f"Value Modifier: {v_modifier}")
    plt.figtext(0.5, 0.1, modifiers, ha="center", fontsize=12)

    plt.gcf().canvas.manager.set_window_title("RGB-HSV-RGB Conversion")
    plt.show()

    cv2.imwrite(f"demo_output/rgb-hsv-converter/original_rgb.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imwrite(f"demo_output/rgb-hsv-converter/rgb_roundtrip_mod-{h_modifier}-{s_modifier}-{v_modifier}.png",
                cv2.cvtColor(rgb_roundtrip_img, cv2.COLOR_BGR2RGB))



def main(argc: int, argv: list):
    if argc != 2:
        raise ValueError("Invalid syntax\n"
                         "python DEMO_SCRIPT.py <image_filename>")

    try:
        filename: str = argv[1]
    except TypeError:
        raise TypeError("Invalid data type entered. Following are the datatypes:\n"
                        "Filename : String\n"
                        "Height : int\n")

    image = cv2.imread(filename)
    if image is None:
        raise FileNotFoundError(f"Could not load {filename}. Check your path.")

    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_dir = "demo_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # RGB to HSV to RGB - 0, 0, 0 as modifiers
    demo_rgb_hsv_rgb_conversion(rgb_img)

    # RGB to HSV to RGB - 0, 0.5, 0 as modifiers
    demo_rgb_hsv_rgb_conversion(rgb_img, 0, 0.5, 0)

    # RGB to HSV to RGB - 0, 0.5, 0.5 as modifiers
    demo_rgb_hsv_rgb_conversion(rgb_img, 0, 0.5, 0.5)

    # RGB to HSV to RGB - 180, 0, 0 as modifiers
    demo_rgb_hsv_rgb_conversion(rgb_img, 180, 0, 0)



if __name__ == "__main__":
    import sys

    main(len(sys.argv), sys.argv)
