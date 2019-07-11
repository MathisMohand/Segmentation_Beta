from keras.models import Model
from keras.layers import *
from keras.optimizers import *


def model():

    inputs = Input((224, 224, 3))

    conv1 = Conv2D(64, 3, padding='same', activation='relu')(inputs)
    conv2 = Conv2D(64, 3, padding='same', activation='relu')(conv1)
    pool1 = MaxPool2D()(conv2)

    conv3 = Conv2D(128, 3, padding='same', activation='relu')(pool1)
    conv4 = Conv2D(128, 3, padding='same', activation='relu')(conv3)
    pool2 = MaxPool2D()(conv4)

    conv5 = Conv2D(256, 3, padding='same', activation='relu')(pool2)
    conv6 = Conv2D(256, 3, padding='same', activation='relu')(conv5)
    pool3 = MaxPool2D()(conv6)

    conv7 = Conv2D(512, 3, padding='same', activation='relu')(pool3)
    conv8 = Conv2D(512, 3, padding='same', activation='relu')(conv7)
    pool4 = MaxPool2D()(conv8)

    conv9 = Conv2D(1024, 3, padding='same', activation='relu')(pool4)
    conv10 = Conv2D(1024, 3, padding='same', activation='relu')(conv9)
    up_samp1 = UpSampling2D()(conv10)
    up_conv1 = Conv2D(512, 2, padding='same', activation='relu')(up_samp1)

    merge1 = concatenate([conv8, up_conv1], axis=3)
    conv11 = Conv2D(512, 3, padding='same', activation='relu')(merge1)
    conv12 = Conv2D(512, 3, padding='same', activation='relu')(conv11)
    up_samp2 = UpSampling2D()(conv12)
    up_conv2 = Conv2D(256, 2, padding='same', activation='relu')(up_samp2)

    merge2 = concatenate([conv6, up_conv2], axis=3)
    conv13 = Conv2D(256, 3, padding='same', activation='relu')(merge2)
    conv14 = Conv2D(256, 3, padding='same', activation='relu')(conv13)
    up_samp3 = UpSampling2D()(conv14)
    up_conv3 = Conv2D(128, 2, padding='same', activation='relu')(up_samp3)

    merge3 = concatenate([conv4, up_conv3], axis=3)
    conv15 = Conv2D(128, 3, padding='same', activation='relu')(merge3)
    conv16 = Conv2D(128, 3, padding='same', activation='relu')(conv15)
    up_samp4 = UpSampling2D()(conv16)
    up_conv4 = Conv2D(64, 2, padding='same', activation='relu')(up_samp4)

    merge4 = concatenate([conv2, up_conv4], axis=3)
    conv17 = Conv2D(64, 3, padding='same', activation='relu')(merge4)
    conv18 = Conv2D(64, 3, padding='same', activation='relu')(conv17)
    conv19 = Conv2D(2, 1, padding='same', activation='softmax')(conv18)

    model = Model(inputs=inputs, outputs=conv19)

    model.compile(optimizer=Adagrad(), loss='categorical_crossentropy', metrics=['mae', 'acc'])


if __name__ == '__main__':
    model()
