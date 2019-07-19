import os

import numpy as np
from segmentation.DataGenerator import DataGenerator
from segmentation.unet import unet_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from PIL import Image

TRAIN_PATH = "../images/train/"
DATA_TRAIN = "../data/train/"
DATA_VALID = "../data/valid/"
VALID_PATH = "../images/valid/"
TEST_PATH = "../images/test/"
DATA_TEST = "../data/test"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def predict(directory, weights):

    model = unet_model()
    model.load_weights(weights)
    ls = os.listdir(directory)

    test = []

    for f_name in ls:
        test.append(np.array(Image.open(directory + f_name)))

    results = model.predict(np.array(test) / 255., 1, verbose=1)

    i = 0
    for f_name in ls:
        tmp = results[i, :, :, 0] * 255.
        img = Image.fromarray(tmp)
        img.save(directory + f_name)
        i += 1


def main():

    model = unet_model()
    batch_size = 4

    train_ids = next(os.walk(DATA_TRAIN + "images/"))[2]
    valid_ids = next(os.walk(DATA_VALID + "images/"))[2]

    train_gen = DataGenerator(train_ids, DATA_TRAIN, batch_size=batch_size, image_size=IMAGE_WIDTH)
    valid_gen = DataGenerator(valid_ids, DATA_VALID, batch_size=batch_size, image_size=IMAGE_WIDTH)

    train_steps = len(train_ids) // batch_size
    valid_steps = len(valid_ids) // batch_size

    model_checkpoint = ModelCheckpoint('unet_cp_3.h5', monitor='val_loss', verbose=1, save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', verbose=1,
                                   min_delta=0.0001, patience=15, mode='min')

    callbacks = [model_checkpoint, early_stopping]

    model.load_weights("unet_cp1.h5")

    model.fit_generator(generator=train_gen, validation_data=valid_gen, steps_per_epoch=train_steps,
                        validation_steps=valid_steps, callbacks=callbacks,
                        epochs=100, verbose=1, use_multiprocessing=True, workers=4)

    model.save_weights("unet_weights_1.h5")

    predict(TEST_PATH + "images/", "unet_cp1.h5")


if __name__ == "__main__":
    main()
