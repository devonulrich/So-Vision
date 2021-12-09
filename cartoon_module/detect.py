import numpy as np
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt

# can detect faces that are 0.25% of the image by area
# analyzed on iCartoonFace -- should be fine for 90% of faces
# for a typical image, this means scaling by 7x
FACE_SIZE_THRESHOLD = 0.0025

def sliding_window(model, img, doPrint=False):
    pts_x = np.arange(0, img.shape[1] - 224, 32)
    pts_y = np.arange(0, img.shape[0] - 224, 32)
    all_pts_x = np.array([pts_x for _ in pts_y]).ravel()
    all_pts_y = np.array([pts_y for _ in pts_x]).T.ravel()

    hmap = np.zeros(img.shape[:2])
    for i in range(len(all_pts_x)):
        if i % 100 == 0 and doPrint:
            print(i, '/', len(all_pts_x))
        curr_x = all_pts_x[i]
        curr_y = all_pts_y[i]
        patch = img[curr_y : curr_y + 224, curr_x : curr_x + 224, :]
        val = model.predict(patch[np.newaxis, ...])
        # if val > 0.95:
        hmap[curr_y : curr_y + 224, curr_x : curr_x + 224] += val
    plt.imsave('test.png', hmap)

def detect_on_img(model, img : np.ndarray):
    # first scale img up
    min_face_area = img.shape[0] * img.shape[1] * FACE_SIZE_THRESHOLD
    min_face_size = np.sqrt(min_face_area)
    scale_factor = 224 / min_face_size
    newShape = (int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor))
    largeImg = cv2.resize(img, newShape)

    sliding_window(model, largeImg, doPrint=True)


def main():
    model = keras.models.load_model('./modelout')
    testImg = plt.imread('/Users/devon/Desktop/personai_icartoonface_dettrain/icartoonface_dettrain/personai_icartoonface_dettrain_22378.jpg')
    detect_on_img(model, testImg)
    plt.waitforbuttonpress()

if __name__ == '__main__':
    main()