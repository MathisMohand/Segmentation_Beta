import os

import numpy as np

from segmentation.unet import unet_model

TRAIN_PATH = "../data/train/"
VALID_PATH = "../data/valid/"
NPY = ".npy"


def generate_data(directory, batch_size):

    i = 0
    # All images data, either in train / test or valid
    ls = os.listdir(directory)

    # Rather shuffling indexes than files themselves as we need same id for labels
    rand_idx = np.arange(len(ls) / 2, dtype=int)
    np.random.shuffle(rand_idx)

    while True:
        image_batch = []
        label_batch = []
        for b in range(batch_size):
            if i == len(ls) / 2:
                i = 0
                np.random.shuffle(rand_idx)

            x_sample = np.load(directory + str(rand_idx[i]) + NPY)
            y_sample = np.load(directory + str(rand_idx[i]) + "m" + NPY)
            i += 1

            # Loading the npy file which is already an array
            x = x_sample.reshape((224, 224, 3))
            y = y_sample.reshape((224, 224, 1))

            # Normalization
            image_batch.append(x.astype(float)/255)
            label_batch.append(y)

        yield (np.array(image_batch), np.array(label_batch))


def main():

    model = unet_model()
    batch_size = 1
    length = len(os.listdir(TRAIN_PATH)) / 2
    valid_length = len(os.listdir(VALID_PATH)) / 2

    model.fit_generator(generate_data(TRAIN_PATH, batch_size), steps_per_epoch=length/batch_size, epochs=10, verbose=2,
                        validation_data=generate_data(VALID_PATH, batch_size), validation_steps=valid_length/batch_size,
                        use_multiprocessing=False)

    model.save_weights("unet-weights-10")


if __name__ == "__main__":
    main()
