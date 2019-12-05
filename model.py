import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as keras


def unet(input_size=(64, 80, 1), num_classes=10, use_sep_conv=False, use_deconv=False):
    inputs = Input(input_size)
    if use_sep_conv:
        conv1 = Conv2D(8, 1, padding='same')(inputs)
        conv1 = Conv2D(16, 1, padding='same',
                       kernel_initializer='he_normal')(DepthwiseConv2D(3,
                                                                       padding='same',
                                                                       kernel_initializer='he_normal')(conv1))
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Conv2D(16, 1, padding='same',
                       kernel_initializer='he_normal')(DepthwiseConv2D(3,
                                                                       padding='same',
                                                                       kernel_initializer='he_normal')(conv1))
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
    else:
        conv1 = Conv2D(8, 3, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Conv2D(8, 3, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    if use_sep_conv:
        conv2 = Conv2D(20, 1, padding='same',
                       kernel_initializer='he_normal')(DepthwiseConv2D(3,
                                                                       padding='same',
                                                                       kernel_initializer='he_normal')(pool1))
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Conv2D(20, 1, padding='same',
                       kernel_initializer='he_normal')(DepthwiseConv2D(3,
                                                                       padding='same',
                                                                       kernel_initializer='he_normal')(conv2))
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
    else:
        conv2 = Conv2D(12, 3, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Conv2D(12, 3, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    if use_sep_conv:
        conv3 = Conv2D(32, 1, padding='same',
                       kernel_initializer='he_normal')(DepthwiseConv2D(3,
                                                                       padding='same',
                                                                       kernel_initializer='he_normal')(pool2))
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv2D(32, 1, padding='same',
                       kernel_initializer='he_normal')(DepthwiseConv2D(3,
                                                                       padding='same',
                                                                       kernel_initializer='he_normal')(conv3))
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
    else:
        conv3 = Conv2D(16, 3, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv2D(16, 3, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    if use_sep_conv:
        conv4 = Conv2D(32, 1, padding='same',
                       kernel_initializer='he_normal')(DepthwiseConv2D(3,
                                                                       padding='same',
                                                                       kernel_initializer='he_normal')(pool3))
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = Conv2D(32, 1, padding='same',
                       kernel_initializer='he_normal')(DepthwiseConv2D(3,
                                                                       padding='same',
                                                                       kernel_initializer='he_normal')(conv4))
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
    else:
        conv4 = Conv2D(24, 3, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = Conv2D(24, 3, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
    if use_sep_conv:
        up5 = Conv2D(48, 1, padding='same', kernel_initializer='he_normal')(DepthwiseConv2D(
            3, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2),
                                                                            interpolation='bilinear')(conv4)))

    elif use_deconv:
        up5 = Conv2DTranspose(48, 3, 2, activation='relu', padding='same', kernel_initializer='he_normal')((conv4))
    else:
        up5 = Conv2D(24, 3, activation='relu', padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2, 2), interpolation='bilinear')(conv4))
    up5 = BatchNormalization()(up5)
    up5 = Activation('relu')(up5)
    merge5 = Concatenate(axis=3)([conv3, up5])
    conv5 = Conv2D(24, 3, padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(24, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    if use_sep_conv:
        up6 = Conv2D(36, 1, padding='same', kernel_initializer='he_normal')(DepthwiseConv2D(
            3, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2),
                                                                            interpolation='bilinear')(conv5)))

    elif use_deconv:
        up6 = Conv2DTranspose(16, 3, 2, padding='same', kernel_initializer='he_normal')((conv5))
    else:
        up6 = Conv2D(16, 3, padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5))
    up6 = BatchNormalization()(up6)
    up6 = Activation('relu')(up6)
    merge6 = Concatenate(axis=3)([conv2, up6])
    conv6 = Conv2D(16, 3, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(16, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    if use_sep_conv:
        up7 = Conv2D(24, 1, padding='same', kernel_initializer='he_normal')(DepthwiseConv2D(
            3, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2),
                                                                            interpolation='bilinear')(conv6)))

    elif use_deconv:
        up7 = Conv2DTranspose(12, 3, 2, padding='same', kernel_initializer='he_normal')((conv6))
    else:
        up7 = Conv2D(12, 3, padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6))
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)
    merge7 = Concatenate(axis=3)([conv1, up7])
    conv7 = Conv2D(12, 3, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(12, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv8 = Conv2D(num_classes, 1, activation='softmax')(conv7)

    model = Model(inputs=inputs, outputs=conv8)

    return model