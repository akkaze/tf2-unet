from __future__ import print_function
import numpy as np
import os
import glob
import cv2
import sys
from tqdm import tqdm
import time
np.set_printoptions(threshold=sys.maxsize)

# def letterbox_resize(img, expected_size, interp):
#     ih, iw = img.shape[0:2]
#     ew, eh = expected_size
#     scale = min(eh / ih, ew / iw)
#     nh = int(ih * scale)
#     nw = int(iw * scale)
#     smat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
#     top = (eh - nh) // 2
#     bottom = eh - nh - top
#     left = (ew - nw) // 2
#     right = ew - nw - left
#     tmat = np.array([[1, 0, left], [0, 1, top], [0, 0, 1]], np.float32)
#     amat = np.dot(tmat, smat)
#     amat_ = amat[:2, :]
#     dst = cv2.warpAffine(img, amat_, expected_size, flags=interp)
#     if dst.ndim == 2:
#         dst = np.expand_dims(dst, axis=-1)
#     return dst


def letterbox_resize(image, expected_size, interp, show=False):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=interp)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    dst = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    if dst.ndim == 2:
        dst = np.expand_dims(dst, axis=-1)
    return dst


def batched_letterbox_resize(imgs, expected_size, interp, show=False):
    assert imgs.ndim == 4
    resized_imgs = []
    for i in range(imgs.shape[0]):
        img = np.squeeze(imgs[i:i + 1], axis=0)
        rsz_imgs = []
        for j in range(imgs.shape[-1]):
            tmp_img = img[..., j:j + 1]

            tmp_img = letterbox_resize(tmp_img, expected_size, interp, show)
            rsz_imgs.append(tmp_img)
        img = np.concatenate(rsz_imgs, axis=-1)
        resized_imgs.append(img)
    imgs = np.stack(resized_imgs)
    del resized_imgs
    return imgs


def m2nist(data_rootdir, dst_size=(80, 64), val_ratio=0.2):
    imgs = np.load(os.path.join(data_rootdir, 'combined.npy')).astype(np.uint8)
    # imgs = np.expand_dims(imgs, axis=-1)
    # imgs = batched_letterbox_resize(imgs, dst_size, cv2.INTER_CUBIC)
    # print(imgs.shape)
    #np.save('combined.npy', imgs)
    masks = np.load(os.path.join(data_rootdir, 'segmented.npy')).astype(np.uint8)
    # masks = masks[..., 0:10]
    # masks = batched_letterbox_resize(masks, dst_size, cv2.INTER_NEAREST, True)
    # print(masks.shape)
    #np.save('segmented.npy', masks)
    num_data = imgs.shape[0]
    num_val = int(num_data * val_ratio)
    num_train = num_data - num_val
    train_imgs, val_imgs = imgs[:num_train, ...], imgs[num_train:, ...]
    train_masks, val_masks = masks[:num_train, ...], masks[num_train:, ...]
    return (train_imgs.astype(np.float32) / 255., train_masks.astype(np.int32)), (val_imgs.astype(np.float32) / 255.,
                                                                                  val_masks.astype(np.int32))