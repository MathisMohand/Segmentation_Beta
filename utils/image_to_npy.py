import numpy as np
from PIL import Image


DATA_PATH = "../data/"
IMAGES_PATH = "../images/"
IMG_PATH = "images/"
MSK_PATH = "masks/"
TIF = ".tif"
NPY = ".npy"


def get_image_array(name):
    img = Image.open(IMAGES_PATH + name + TIF, "r")
    pixels = np.array(img)
    return pixels


def image_to_npy(name, folder):
    tab = get_image_array(folder + name)
    dest = DATA_PATH + folder + name + NPY
    np.save(dest, tab)


def main():
    folders = ["train/", "test/", "valid/"]
    sizes = [0, 600, 800, 1000]

    for j in range(len(folders)):
        for i in range(sizes[j], sizes[j+1]):
            image_to_npy(str(i), folders[j] + IMG_PATH)
            image_to_npy(str(i) + "m", folders[j] + MSK_PATH)


if __name__ == '__main__':
    main()
