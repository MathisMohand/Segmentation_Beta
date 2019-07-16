import os

import numpy as np
from keras.callbacks import ModelCheckpoint

from segmentation.unet import unet_model

from PIL import Image

TRAIN_PATH = "../data/train/"
VALID_PATH = "../data/valid/"
DUMMY_PATH = "../data/dummy/"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
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
            if directory == VALID_PATH:
                x_sample = np.load(directory + str(rand_idx[i] + 800) + NPY)
                y_sample = np.load(directory + str(rand_idx[i] + 800) + "m" + NPY)
            else:
                x_sample = np.load(directory + str(rand_idx[i]) + NPY)
                y_sample = np.load(directory + str(rand_idx[i]) + "m" + NPY)
            i += 1

            # Loading the npy file which is already an array
            x = x_sample.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 3))
            y = y_sample.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 1))
            # y_neg = 255 - y
            # y_fin = np.array([y, y_neg]).reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 2))

            # Normalization
            image_batch.append(x.astype(float)/255)
            label_batch.append(y.astype(float)/255)

        yield (np.array(image_batch), np.array(label_batch))


def predict(directory, weights):

    model = unet_model()
    model.load_weights(weights)
    ls = os.listdir(directory)

    data = []

    for i in range(len(ls)):
        data[i] = np.load(directory + str(i) + NPY)

    results = model.predict(data, 1, verbose=1)

    for i in range(len(results)):
        img = Image.fromarray(results[i, :, :, 0] * 255)
        img.save(directory + str(i) + "r.png")


def main():

    model = unet_model()
    batch_size = 4
    length = len(os.listdir(DUMMY_PATH)) / 2
    valid_length = len(os.listdir(VALID_PATH)) / 2

    model_checkpoint = ModelCheckpoint('unet_test.h5', monitor='loss', verbose=1, save_best_only=True)

    model.load_weights("unet_test.h5")

    # model.summary()

    model.fit_generator(generate_data(DUMMY_PATH, batch_size), steps_per_epoch=length / batch_size, epochs=20,
                        verbose=2,
                        validation_data=generate_data(VALID_PATH, batch_size),
                        validation_steps=valid_length / batch_size,
                        use_multiprocessing=True,
                        callbacks=[model_checkpoint])

    model.save_weights("unet2-dummy_v1.h5")

    test = np.ones((52, 256, 256, 3))

    for i in range(0, 52):
        test[i] = np.load("../data/dummy/" + str(i) + NPY)

    results = model.predict(test, 1, verbose=1)

    for i in range(len(results)):
        tmp = results[i, :, :, 0] * 255.
        print(np.unique(results[i, :, :, 0]))
        print(np.unique(tmp))
        print(results.shape)
        img = Image.fromarray(tmp)
        img.save("../images/dummy/" + str(i) + "r.tif")


if __name__ == "__main__":
    main()
