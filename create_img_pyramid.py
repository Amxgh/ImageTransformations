import cv2
import numpy as np

from img_transforms import resize_img


def create_img_pyramid(img: np.ndarray, height: int) -> list[np.ndarray]:
    """
    Create an image pyramid by resizing the input image to different scales.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W, C), where H is the height, W is the width, and C is the number of channels.
    height : float
        The number of scales to resize the image to.

    Returns
    -------
    list[np.ndarray]
        List of resized images, where each image has shape (new_H, new_W, C), with the same dtype as the input.
    """
    h, w, c = img.shape
    resized_imgs = []
    for i in range(height):
        resized_img = resize_img(img, 0.5 ** (i + 1))
        resized_imgs.append(resized_img)
    return resized_imgs


def main(argc: int, argv: list):
    if argc != 3:
        raise ValueError("Invalid syntax\n"
                         "python create_img_pyramid.py <filename> <height>")

    try:
        filename: str = argv[1]
        height: int = int(argv[2])
    except TypeError:
        raise TypeError("Invalid data type entered. Following are the datatypes:\n"
                        "Filename : String\n"
                        "Height : int\n")

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_imgs = create_img_pyramid(img, height)

    for i in range(height):
        cv2.imwrite(f"{filename[:-4]}_{2 ** (i + 1)}x.png", cv2.cvtColor(resized_imgs[i], cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    import sys

    main(len(sys.argv), sys.argv)
