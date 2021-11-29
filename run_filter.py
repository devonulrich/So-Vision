import cv2
from PIL import Image
import numpy as np

from mtcnn import MTCNN
from face_filters_video import get_persp_face_pts

import os

detector = MTCNN()

count = 0

pe_filt = cv2.imread('pe_glasses.png', cv2.IMREAD_UNCHANGED)
filt_pts = np.array([[31, 31], [95, 31], [31, 127], [95, 127]], dtype=np.float32)

def apply_filter(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detOut = detector.detect_faces(imgRGB)

    imgPIL = Image.fromarray(img).convert('RGBA')
    for face in detOut:
        persp_pts = np.round(get_persp_face_pts(face))
        M = cv2.getPerspectiveTransform(filt_pts, persp_pts)
        filtOut = cv2.warpPerspective(pe_filt, M, (img.shape[1], img.shape[0]))

        filtPIL = Image.fromarray(filtOut)
        imgPIL.alpha_composite(filtPIL)
    
    return np.asarray(imgPIL)


def process_files(f, out_dir, prefix=''):
    global count

    if os.path.isdir(f):
        # descend directory
        for subfile in os.listdir(f):
            process_files(f + '/' + subfile, out_dir, prefix)
        return

    img = cv2.imread(f)
    result = apply_filter(img)

    out_file = out_dir + '/' + prefix + str(count).zfill(5) + '.jpg'
    cv2.imwrite(out_file, result)
    count += 1

    
if __name__ == '__main__':
    path = '.'
    if os.path.isfile('./dataset_path'):
        with open('./dataset_path') as f:
            path = f.readline()[:-1]

    print('Starting...')
    os.system('mkdir ' + path + '/out')
    process_files(path + '/fddb', path + '/out', prefix='fddb')