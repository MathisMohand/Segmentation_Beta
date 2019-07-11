from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = 1000000000
DENVER_PATH = "../denver_files/"
TRAIN_PATH = "../images/train/"
TEST_PATH = "../images/test/"
VALID_PATH = "../images/valid/"
PATCH_SIZE = 224


def load_image(path):
    image = Image.open(path)
    pixels = np.array(image)

    return pixels


def binarize_image(img):
    bin_img = np.zeros(img.shape)
    bin_img[img > 0] = 255

    return bin_img


def save_image(img_array, path):
    image = Image.fromarray(img_array)
    image.save(path)


def create_mask(src_raster, dest_raster):
    img = load_image(src_raster)
    mask = binarize_image(img)
    save_image(mask, dest_raster)


def generate_patch(image, mask):
    rand_x = np.random.randint(image.shape[0] - PATCH_SIZE)
    rand_y = np.random.randint(image.shape[1] - PATCH_SIZE)
    patch = image[rand_x:rand_x + PATCH_SIZE, rand_y:rand_y + PATCH_SIZE]
    patch_mask = mask[rand_x:rand_x + PATCH_SIZE, rand_y:rand_y + PATCH_SIZE]

    return patch, patch_mask


def generate_dataset(img_path, mask_path):
    image = load_image(img_path)
    mask = load_image(mask_path)

    for i in range(600):
        patch, patch_mask = generate_patch(image, mask)
        save_image(patch, TRAIN_PATH + str(i) + ".tif")
        save_image(patch_mask, TRAIN_PATH + str(i) + "m.tif")

    for i in range(600, 800):
        patch, patch_mask = generate_patch(image, mask)
        save_image(patch, TEST_PATH + str(i) + ".tif")
        save_image(patch_mask, TEST_PATH + str(i) + "m.tif")

    for i in range(800, 1000):
        patch, patch_mask = generate_patch(image, mask)
        save_image(patch, VALID_PATH + str(i) + ".tif")
        save_image(patch_mask, VALID_PATH + str(i) + "m.tif")


# TODO Data Augmentation
# To be done after segmentation


def main():
    # path = DENVER_PATH + 'raster_test_1.tif'
    # create_mask(path, DENVER_PATH + "mask_1.tif")

    image_path = DENVER_PATH + "Denver_rectified.tif"
    mask_path = DENVER_PATH + "mask_1.tif"
    generate_dataset(image_path, mask_path)


if __name__ == '__main__':
    main()
