import os

import numpy as np
from segmentation.DataGenerator import DataGenerator
from segmentation.unet import unet_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from PIL import Image

TRAIN_PATH = "../images/train/"
VALID_PATH = "../images/valid/"
DUMMY_PATH = "../images/dummy/"
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
                x_sample = np.load(directory + "images/" + str(rand_idx[i] + 800) + NPY)
                y_sample = np.load(directory + "masks/" + str(rand_idx[i] + 800) + "m" + NPY)
            else:
                x_sample = np.load(directory + "images/" + str(rand_idx[i]) + NPY)
                y_sample = np.load(directory + "masks/" + str(rand_idx[i]) + "m" + NPY)
            i += 1

            # Loading the npy file which is already an array
            x = x_sample.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 3))
            y = y_sample.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 1))
            # y_neg = 255 - y
            # y_fin = np.array([y, y_neg]).reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 2))

            # Normalization
            image_batch.append(x.astype(float)/255)
            label_batch.append(y.astype(float)/255)

        yield np.array(image_batch), np.array(label_batch)


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
    batch_size = 32

    train_ids = next(os.walk(TRAIN_PATH + "images/"))[2]
    valid_ids = next(os.walk(VALID_PATH + "images/"))[2]
    # dummy_ids = next(os.walk(DUMMY_PATH + "images/"))[2]

    train_gen = DataGenerator(train_ids, TRAIN_PATH, batch_size=batch_size, image_size=IMAGE_WIDTH)
    valid_gen = DataGenerator(valid_ids, VALID_PATH, batch_size=batch_size, image_size=IMAGE_WIDTH)
    # dummy_gen = DataGenerator(dummy_ids, DUMMY_PATH, batch_size=batch_size, image_size=IMAGE_WIDTH)
    # dummy_gen2 = DataGenerator(dummy_ids, DUMMY_PATH, batch_size=batch_size, image_size=IMAGE_WIDTH)

    train_steps = len(train_ids) // batch_size
    valid_steps = len(valid_ids) // batch_size
    # dummy_steps = len(dummy_ids) // batch_size

    model_checkpoint = ModelCheckpoint('unet_cp1.h5', monitor='val_loss', verbose=1, save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', verbose=1,
                                   min_delta=0.001, patience=150, mode='min')

    model.load_weights("unet_test2.h5")

    # model.summary()

    model.fit_generator(generator=train_gen, validation_data=valid_gen, steps_per_epoch=train_steps,
                        validation_steps=valid_steps, callbacks=[model_checkpoint, early_stopping],
                        epochs=1000, verbose=1, use_multiprocessing=True, workers=8)

    model.save_weights("unet_test10k.h5")

    test = np.ones((1000, 256, 256, 3))

    for i in range(0, 1000):
        test[i] = Image.open("../images/test/images/" + str(i + 8000) + ".tif")

    results = model.predict(np.array(test), 1, verbose=1)

    for i in range(len(results)):
        tmp = results[i, :, :, 0] * 255.
        img = Image.fromarray(tmp)
        img.save("../images/test/masks/" + str(i + 8000) + "r.tif")


if __name__ == "__main__":
    main()
