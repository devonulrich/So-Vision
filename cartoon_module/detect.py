import numpy as np
import tensorflow as tf
from tensorflow import keras

import cv2
import matplotlib.pyplot as plt
import pickle

import util

# size of each input window. based on network ; should not be changed
WINDOW_SIZE = 224

# can detect faces that are 0.25% of the image by area
# analyzed on iCartoonFace -- should be fine for 90% of faces
# for a typical image, this means scaling by 7x
FACE_SIZE_THRESHOLD = 0.0025
# min confidence for a rectangle in the sliding window to be considered a proposal
MIN_CONFIDENCE = 0.2
# new_size = size * RESCALING_FACTOR; 0.8 is roughly 3 images per octave
RESCALING_FACTOR = 0.8
# from paper
NMS_THRESHOLD = 0.2

def sliding_window(model, img, doPrint=False):
    pts_x = np.arange(0, img.shape[1] - WINDOW_SIZE, 32) # stride = 32 
    pts_y = np.arange(0, img.shape[0] - WINDOW_SIZE, 32)
    all_pts_x = np.array([pts_x for _ in pts_y]).ravel()
    all_pts_y = np.array([pts_y for _ in pts_x]).T.ravel()

    rects = []
    rects_conf = []
    for i in range(len(all_pts_x)):
        if i % 100 == 0 and doPrint:
            print(i, '/', len(all_pts_x))
        curr_x = all_pts_x[i]
        curr_y = all_pts_y[i]
        patch = img[curr_y : curr_y + WINDOW_SIZE, curr_x : curr_x + WINDOW_SIZE, :]
        conf = model.predict(patch[np.newaxis, ...])[0,0]
        if conf > MIN_CONFIDENCE:
            rects.append([curr_x, curr_y, WINDOW_SIZE, WINDOW_SIZE])
            rects_conf.append(conf)
    # plt.imsave('test.png', hmap)
    return np.array(rects), np.array(rects_conf)

def NMS_Max(rects, rects_conf):
    final_rects = []
    final_confs = []
    idxs = np.argsort(rects_conf)
    for i in range(len(rects)):
        # curr rect
        idx_i = idxs[i]
        include = True
        for j in range(i+1, len(rects)):
            # comparison rect
            idx_j = idxs[j]
            if util.IoU(rects[idx_i], rects[idx_j]) > NMS_THRESHOLD:
                include = False
                break
        if include:
            final_rects.append(rects[idx_i])
            final_confs.append(rects_conf[idx_i])

    return np.array(final_rects), np.array(final_confs)

def Soft_NMS(rects, rects_conf):
    rects_conf = rects_conf.copy()
    final_rects = []
    final_confs = []
    idxs = np.argsort(rects_conf)
    for i in range(len(rects)):
        # curr rect (low score)
        idx_i = idxs[i]
        for j in range(i+1, len(rects)):
            # comparison rect
            idx_j = idxs[j]
            if util.IoU(rects[idx_i], rects[idx_j]) > NMS_THRESHOLD:
                rects_conf[idx_i] = rects_conf[idx_i] * (1 - util.IoU(rects[idx_i], rects[idx_j]))

        # final thresholding
        if rects_conf[idx_i] > 0.5:
            final_rects.append(rects[idx_i])
            final_confs.append(rects_conf[idx_i])

    return np.array(final_rects), np.array(final_confs)

def detect_on_img(model, img : np.ndarray, soft=False):
    # first scale img up
    min_face_area = img.shape[0] * img.shape[1] * FACE_SIZE_THRESHOLD
    min_face_size = np.sqrt(min_face_area)
    scale_factor = WINDOW_SIZE / min_face_size
    newShape = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    img = cv2.resize(img, newShape)

    print('starting sliding window')
    all_rects = np.empty((0, 4))
    all_confs = np.empty((0))
    while(img.shape[0] >= WINDOW_SIZE and img.shape[1] >= WINDOW_SIZE):
        print('scale:', scale_factor)
        print('dims:', img.shape)
        rects, rects_conf = sliding_window(model, img)
        if len(rects) > 0:
            rects = rects / scale_factor
            all_rects = np.vstack((all_rects, rects))
            all_confs = np.concatenate((all_confs, rects_conf))

        scale_factor = scale_factor * RESCALING_FACTOR
        newShape = (int(img.shape[1] * RESCALING_FACTOR), int(img.shape[0] * RESCALING_FACTOR))
        img = cv2.resize(img, newShape)

    if soft:
        final_rects, final_confs = Soft_NMS(all_rects, all_confs)
    else:
        final_rects, final_confs = NMS_Max(all_rects, all_confs)
    return final_rects, final_confs

def main():
    model = keras.models.load_model('modelout')
    testImgPath = '/scratch/network/dulrich/personai_icartoonface_dettrain/icartoonface_dettrain/personai_icartoonface_dettrain_22378.jpg'
    testImg = cv2.cvtColor(cv2.imread(testImgPath), cv2.COLOR_BGR2RGB)
    detect_on_img(model, testImg)

if __name__ == '__main__':
    main()
