from PIL import Image
import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):

    def __init__(self, ids, path, batch_size=4, image_size=256):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __load__(self, idf):
        image_path = self.path + 'images/' + idf
        mask_path = self.path + 'masks/' + idf[:-4] + 'm.tif'

        image = Image.open(image_path)
        image.resize((self.image_size, self.image_size))

        mask = Image.open(mask_path)
        mask.resize((self.image_size, self.image_size))

        image = np.array(image) / 255.
        mask = np.reshape(np.array(mask) / 255., (self.image_size, self.image_size, 1))

        return image, mask

    def __loadnpy__(self, idf):
        image_path = self.path + 'images/' + idf
        mask_path = self.path + 'masks/' + idf[:-4] + 'm.npy'

        image = np.load(image_path).reshape((self.image_size, self.image_size, 3))
        mask = np.load(mask_path).reshape((self.image_size, self.image_size, 1))

        image = image / 255.
        mask = mask / 255.

        return image, mask

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size

        files_batch = self.ids[index * self.batch_size:(index + 1) * self.batch_size]

        image = []
        mask = []

        for idf in files_batch:
            _image, _mask = self.__loadnpy__(idf)
            image.append(_image)
            mask.append(_mask)

        image = np.array(image)
        mask = np.array(mask)

        return image, mask

    def on_epoch_end(self):
        np.random.shuffle(self.ids)

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))
