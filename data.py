from __future__ import print_function
import numpy as np
import os
import glob
import cv2
import sys
import time
import tensorflow as tf
import requests
import zipfile
from six.moves import urllib
from tqdm import tqdm


def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(urllib.request.urlopen(url).info().get('Content-Length', -1))
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    print(file_size)
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(total=file_size, initial=first_byte, unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with (open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size


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


def download_m2nist_if_not_exist():
    data_rootdir = os.path.expanduser('~/.m2nist')
    m2nist_zip_path = os.path.join(data_rootdir, 'm2nist.zip')
    if os.path.exists(m2nist_zip_path):
        return
    os.makedirs(data_rootdir, exist_ok=True)
    m2nist_zip_url = 'https://raw.githubusercontent.com/akkaze/datasets/master/m2nist.zip'
    download_from_url(m2nist_zip_url, m2nist_zip_path)
    zipf = zipfile.ZipFile(m2nist_zip_path)
    zipf.extractall(data_rootdir)
    zipf.close()


def m2nist(dst_size=(80, 64), batch_size=32, val_ratio=0.2):
    download_m2nist_if_not_exist()
    data_rootdir = os.path.expanduser('~/.m2nist')
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
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.repeat()
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.repeat()
    return train_dataset, val_dataset