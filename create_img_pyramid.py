import numpy as np

from img_transforms import resize_img
import cv2

def create_img_pyramid(filename: str, height: int) -> None:
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    for i in range(height):
        resized_img = resize_img(img, 0.5**(i+1))
        cv2.imwrite(f"{filename.split(".png")[0]}_{(i+1)*2}x.png", cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    return

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

    create_img_pyramid(filename, height)

if __name__ == "__main__":
    import sys
    main(len(sys.argv), sys.argv)