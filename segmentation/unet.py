import keras.models
from keras.layers import *
from keras.optimizers import *


def weighted_cross_entropy(y, y_hat):
    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss_val = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=0.5)

        return tf.reduce_mean(loss_val)

    return loss(y, y_hat)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return coef


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def unet_model():
    inputs = Input((256, 256, 3))

    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu')(inputs)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv1)
    pool1 = MaxPool2D()(conv2)

    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')(pool1)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv3)
    pool2 = MaxPool2D()(conv4)

    conv5 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu')(pool2)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv5)
    pool3 = MaxPool2D()(conv6)

    conv7 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='relu')(pool3)
    conv8 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv7)
    pool4 = MaxPool2D()(conv8)

    conv9 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal', activation='relu')(pool4)
    conv10 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv9)
    up_samp1 = UpSampling2D()(conv10)
    up_conv1 = Conv2D(512, 2, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp1)

    merge1 = concatenate([conv8, up_conv1], axis=3)
    conv11 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='relu')(merge1)
    conv12 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv11)
    up_samp2 = UpSampling2D()(conv12)
    up_conv2 = Conv2D(256, 2, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp2)

    merge2 = concatenate([conv6, up_conv2], axis=3)
    conv13 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu')(merge2)
    conv14 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv13)
    up_samp3 = UpSampling2D()(conv14)
    up_conv3 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp3)

    merge3 = concatenate([conv4, up_conv3], axis=3)
    conv15 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')(merge3)
    conv16 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv15)
    up_samp4 = UpSampling2D()(conv16)
    up_conv4 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp4)

    merge4 = concatenate([conv2, up_conv4], axis=3)
    conv17 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu')(merge4)
    conv18 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv17)
    conv19 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal', activation='relu')(conv18)
    conv20 = Conv2D(1, 1, activation='sigmoid')(conv19)

    model = keras.models.Model(inputs=inputs, outputs=conv20)

    ada = Adam(lr=3e-5)

    model.compile(optimizer=ada, loss=weighted_cross_entropy, metrics=[dice_coef])

    return model


if __name__ == '__main__':
    unet_model()
