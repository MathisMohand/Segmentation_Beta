from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = 10000000000
DENVER_PATH = "../denver_files/"
TRAIN_PATH_IMG = "../images/train/images/"
TEST_PATH_IMG = "../images/test/images/"
VALID_PATH_IMG = "../images/valid/images/"
TRAIN_PATH_MSK = "../images/train/masks/"
TEST_PATH_MSK = "../images/test/masks/"
VALID_PATH_MSK = "../images/valid/masks/"
PATCH_SIZE = 256


def load_image(path):
    image = Image.open(path)
    pixels = np.array(image, dtype=np.uint8)

    return pixels


def binarize_image(img, width, height):
    bin_img = np.zeros((width, height), dtype=np.uint8)
    bin_img[np.sum(img, axis=2) > 0] = 255

    return bin_img


def save_image(img_array, path):
    image = Image.fromarray(img_array)
    image.save(path)


def create_mask(src_raster, dest_raster):
    img = load_image(src_raster)
    width, height, _ = img.shape
    mask = binarize_image(img, width, height)
    save_image(mask, dest_raster)


def generate_patch(image, mask):

    rand_x = np.random.randint(mask.shape[0] - PATCH_SIZE)
    rand_y = np.random.randint(mask.shape[1] / 2 - PATCH_SIZE)
    patch = image[rand_x:rand_x + PATCH_SIZE, rand_y:rand_y + PATCH_SIZE]
    patch_mask = mask[rand_x:rand_x + PATCH_SIZE, rand_y:rand_y + PATCH_SIZE]

    return patch, patch_mask


def generate_dataset(img_path, mask_path):
    image = load_image(img_path)
    mask = load_image(mask_path)

    for i in range(8000):
        patch, patch_mask = generate_patch(image, mask)
        save_image(patch, TRAIN_PATH_IMG + str(i) + ".tif")
        save_image(patch_mask, TRAIN_PATH_MSK + str(i) + "m.tif")

    for i in range(8000, 9000):
        patch, patch_mask = generate_patch(image, mask)
        save_image(patch, TEST_PATH_IMG + str(i) + ".tif")
        save_image(patch_mask, TEST_PATH_MSK + str(i) + "m.tif")

    for i in range(9000, 10000):
        patch, patch_mask = generate_patch(image, mask)
        save_image(patch, VALID_PATH_IMG + str(i) + ".tif")
        save_image(patch_mask, VALID_PATH_MSK + str(i) + "m.tif")


def main():
    # path = DENVER_PATH + 'raster_noalpha.tif'
    # create_mask(path, DENVER_PATH + "mask.tif")

    image_path = DENVER_PATH + "Denver_noalpha.tif"
    mask_path = DENVER_PATH + "mask.tif"
    generate_dataset(image_path, mask_path)


if __name__ == '__main__':
    main()
