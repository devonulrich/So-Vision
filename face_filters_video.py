from mtcnn import MTCNN
import cv2
import numpy as np
from matplotlib import pyplot as plt

detector = MTCNN()

vid = cv2.VideoCapture(0)

en_filt = cv2.imread('en_glasses.png', cv2.IMREAD_UNCHANGED)
filt_pts = np.array([[31, 31], [63, 95], [95, 31]], dtype=np.float32)

while(True):
    ret, frame = vid.read()

    frameAlt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = detector.detect_faces(frameAlt)

    for face in out:
        pts = list(face['keypoints'].values())
        img_pts = np.ones((3,2), dtype=np.float32)
        img_pts[0, :] = pts[0] # left eye
        img_pts[1, :] = pts[2] # nose
        img_pts[2, :] = pts[1] # right eye

        M = cv2.getAffineTransform(filt_pts, img_pts)
        filtOut = cv2.warpAffine(en_filt, M, (frame.shape[1], frame.shape[0]))
        mask = filtOut[:,:,3] != 0

        frame[mask] = np.array([0, 0, 0])
        frame += filtOut[:,:,:3]

        # for (x, y) in pts:
        #     cv2.circle(frame, (x, y), 5, (255, 0, 0))


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()