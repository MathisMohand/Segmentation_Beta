import keras.models
from keras.layers import *
from keras.optimizers import *


def unet_model():

    inputs = Input((256, 256, 1))

    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu')(inputs)
    dropout1 = Dropout(0.5)(conv1)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu')(dropout1)
    pool1 = MaxPool2D()(conv2)

    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')(pool1)
    dropout2 = Dropout(0.5)(conv3)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')(dropout2)
    pool2 = MaxPool2D()(conv4)

    conv5 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu')(pool2)
    dropout3 = Dropout(0.5)(conv5)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu')(dropout3)
    pool3 = MaxPool2D()(conv6)

    conv7 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='relu')(pool3)
    dropout4 = Dropout(0.5)(conv7)
    conv8 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='relu')(dropout4)
    pool4 = MaxPool2D()(conv8)

    conv9 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal', activation='relu')(pool4)
    dropout5 = Dropout(0.5)(conv9)
    conv10 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal', activation='relu')(dropout5)
    up_samp1 = UpSampling2D()(conv10)
    up_conv1 = Conv2D(512, 2, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp1)

    merge1 = concatenate([conv8, up_conv1], axis=3)
    conv11 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='relu')(merge1)
    dropout6 = Dropout(0.5)(conv11)
    conv12 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='relu')(dropout6)
    up_samp2 = UpSampling2D()(conv12)
    up_conv2 = Conv2D(256, 2, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp2)

    merge2 = concatenate([conv6, up_conv2], axis=3)
    conv13 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu')(merge2)
    dropout7 = Dropout(0.5)(conv13)
    conv14 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu')(dropout7)
    up_samp3 = UpSampling2D()(conv14)
    up_conv3 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp3)

    merge3 = concatenate([conv4, up_conv3], axis=3)
    conv15 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')(merge3)
    dropout8 = Dropout(0.5)(conv15)
    conv16 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')(dropout8)
    up_samp4 = UpSampling2D()(conv16)
    up_conv4 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp4)

    merge4 = concatenate([conv2, up_conv4], axis=3)
    conv17 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu')(merge4)
    conv18 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv17)
    conv19 = Conv2D(2, 1, padding='same', kernel_initializer='he_normal', activation='softmax')(conv18)  # softmax

    model = keras.models.Model(inputs=inputs, outputs=conv19)

    ada = Adam()

    model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['mae', 'acc'])

    return model


if __name__ == '__main__':
    unet_model()
