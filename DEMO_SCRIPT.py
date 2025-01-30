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


def demo_random_crop(img: np.ndarray) -> None:
    h, w, c = img.shape
    size = int(min(h, w) / 10) # Assumed the size to be a 10th of the height or width (the lower of the two)

    if not os.path.exists("demo_output/random-crop"):
        os.makedirs("demo_output/random-crop")

    crop_1 = random_crop(img, size)
    crop_2 = random_crop(img, size)
    crop_3 = random_crop(img, size)
    crop_4 = random_crop(img, size)
    crop_5 = random_crop(img, size)
    crop_6 = random_crop(img, size)

    fig, axes = plt.subplots(3, 3)
    axes[0, 0].axis(False)

    axes[0, 1].imshow(img)
    axes[0, 1].set_title("Original RGB Image (not to scale)")
    axes[0, 1].axis(False)

    # Hide the other two axes in the first row
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(crop_1)
    axes[1, 0].set_title("Crop 1")
    axes[1, 0].axis(False)

    axes[1, 1].imshow(crop_2)
    axes[1, 1].set_title("Crop 2")
    axes[1, 1].axis(False)

    axes[1, 2].imshow(crop_3)
    axes[1, 2].set_title("Crop 3")
    axes[1, 2].axis(False)

    axes[2, 0].imshow(crop_4)
    axes[2, 0].set_title("Crop 4")
    axes[2, 0].axis(False)

    axes[2, 1].imshow(crop_5)
    axes[2, 1].set_title("Crop 5")
    axes[2, 1].axis(False)

    axes[2, 2].imshow(crop_6)
    axes[2, 2].set_title("Crop 6")
    axes[2, 2].axis(False)

    plt.suptitle("Random Crop", fontsize=16)
    plt.gcf().canvas.manager.set_window_title("Random Crop")
    plt.show()

    cv2.imwrite(f"demo_output/random-crop/original_rgb.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    crops = [crop_1, crop_2, crop_3, crop_4, crop_5, crop_6]
    for i in range(len(crops)):
        cv2.imwrite(f"demo_output/random-crop/size_{size}_crop_{i+1}.png", cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB))


def demo_extract_patch(img: np.ndarray, number_of_patches: int) -> None:
    if not os.path.exists("demo_output/extract-patch"):
        os.makedirs("demo_output/extract-patch")

    patches = extract_patch(img, number_of_patches)

    rows, cols = patches.shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(patches[i, j].astype(np.uint8))
            axes[i, j].axis("off")

    plt.suptitle("Extract Patch", fontsize=16)
    plt.gcf().canvas.manager.set_window_title("Extract Patch")
    plt.show()

    cv2.imwrite(f"demo_output/extract-patch/original_rgb.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(len(patches)):
        cv2.imwrite(f"demo_output/extract-patch/patches_{number_of_patches}_patch_{i+1}.png", cv2.cvtColor(patches[i], cv2.COLOR_BGR2RGB))


def demo_resize_image(img: np.ndarray, scale_factor: float) -> None:
    if not os.path.exists("demo_output/resize-image"):
        os.makedirs("demo_output/resize-image")

    resized_image = resize_img(img, scale_factor)

    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(img)
    axes[0].set_title("Original RGB Image")
    axes[0].axis('off')

    axes[1].imshow(resized_image)
    axes[1].set_title(f"Resized Image (Scale Factor: {scale_factor})")
    axes[1].axis('off')

    plt.suptitle("Resize Image", fontsize=16)
    plt.gcf().canvas.manager.set_window_title("Resize Image")
    plt.subplots_adjust(wspace=0.1)

    plt.figtext(0.5, 0.1, "IMAGE SIZE DIFFERENCE IS NOT VISIBLE IN MATPLOTLIB\n"
                          "Please look at ./demo-output/resize-image/", ha="center", fontsize=12)
    cv2.imwrite(f"demo_output/resize-image/original_rgb.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imwrite(f"demo_output/resize-image/resized_image_{scale_factor}.png", cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()

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

    # Random Crop
    demo_random_crop(rgb_img)

    # Extract Patch
    demo_extract_patch(rgb_img, 5)
    demo_extract_patch(rgb_img, 10)

    # Resize Image
    demo_resize_image(rgb_img, 0.5)
    demo_resize_image(rgb_img, 2)


if __name__ == "__main__":
    import sys

    main(len(sys.argv), sys.argv)
