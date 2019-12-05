from __future__ import print_function
import numpy as np
import os
import glob
import cv2
import sys
import time
import tensorflow as tf


def letterbox_resize(img, expected_size, interp):
    ih, iw = img.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    smat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    tmat = np.array([[1, 0, left], [0, 1, top], [0, 0, 1]], np.float32)
    amat = np.dot(tmat, smat)
    amat_ = amat[:2, :]
    dst = cv2.warpAffine(img, amat_, expected_size, flags=interp)
    if dst.ndim == 2:
        dst = np.expand_dims(dst, axis=-1)
    return dst


def transform_target(img, mask, expected_size):
    img = img.numpy()
    mask = mask.numpy()
    expected_size = tuple(expected_size.numpy())
    img = letterbox_resize(img, expected_size, cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.
    mask = tf.keras.utils.to_categorical(mask, 11, np.uint8)
    mask = letterbox_resize(mask, expected_size, cv2.INTER_NEAREST)
    mask = mask.astype(np.float32)
    return img, mask


def m2nist(data_rootdir, dst_size=(80, 64), val_ratio=0.2):
    imgs = np.load(os.path.join(data_rootdir, 'combined.npy')).astype(np.uint8)
    masks = np.load(os.path.join(data_rootdir, 'segmented.npy')).astype(np.uint8)

    num_data = imgs.shape[0]
    num_train = int(num_data * (1 - val_ratio))

    img_dataset = tf.data.Dataset.from_tensor_slices(imgs)
    mask_dataset = tf.data.Dataset.from_tensor_slices(masks)
    dataset = tf.data.Dataset.zip((img_dataset, mask_dataset))

    def tf_transform_target(img, mask):
        img, mask = tf.py_function(func=transform_target, inp=[img, mask, dst_size], Tout=[tf.float32, tf.float32])
        img.set_shape((*dst_size[::-1], 1))
        mask.set_shape((*dst_size[::-1], 11))
        return img, mask

    dataset = dataset.map(lambda x, y: tf_transform_target(x, y))
    train_dataset = dataset.take(num_train)
    val_dataset = dataset.skip(num_train)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.repeat()
    val_dataset = val_dataset.batch(32)
    val_dataset = val_dataset.repeat()
    return train_dataset, val_dataset