from mtcnn import MTCNN
import cv2
import numpy as np
from matplotlib import pyplot as plt

# returns line perpendicular to l that passes through
# the homogeneous point p_h
def perpendicular_line(l, p_h):
    # negative inverse of slope
    a = l[1]
    b = -l[0]

    # dot prod of line and point must be zero, so use that to solve for c
    c = -(a * p_h[0] + b * p_h[1]) / p_h[2]

    return np.array([a, b, c])

def get_persp_face_pts(face):
    # homogeneous coords for left & right eye pts
    le_h = np.ones(3)
    le_h[:2] = face['keypoints']['left_eye']
    re_h = np.ones(3)
    re_h[:2] = face['keypoints']['right_eye']

    # homogeneous coords for left & right mouth pts
    lm_h = np.ones(3)
    lm_h[:2] = face['keypoints']['mouth_left']
    rm_h = np.ones(3)
    rm_h[:2] = face['keypoints']['mouth_right']

    # construct top & bottom lines
    top_line = np.cross(le_h, re_h)
    bot_line = np.cross(lm_h, rm_h)

    # create left & right lines that pass through each eye
    # and are perpendicular to top line
    left_line = perpendicular_line(top_line, le_h)
    right_line = perpendicular_line(top_line, re_h)

    # find bottom corners (intersection bt left/right and bot lines)
    lb_h = np.cross(left_line, bot_line)
    rb_h = np.cross(right_line, bot_line)

    lb_h /= lb_h[2]
    rb_h /= rb_h[2]

    out_mat = np.array([le_h, re_h, lb_h, rb_h], dtype=np.float32)
    return out_mat[:, :2]


def main():
    detector = MTCNN()

    vid = cv2.VideoCapture(0)

    # en_filt = cv2.imread('en_glasses.png', cv2.IMREAD_UNCHANGED)
    pe_filt = cv2.imread('pe_glasses.png', cv2.IMREAD_UNCHANGED)
    filt_pts = np.array([[31, 31], [95, 31], [31, 127], [95, 127]], dtype=np.float32)

    while(True):
        ret, frame = vid.read()

        frameAlt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = detector.detect_faces(frameAlt)

        for face in out:
            persp_pts = np.round(get_persp_face_pts(face))
            M = cv2.getPerspectiveTransform(filt_pts, persp_pts)
            filtOut = cv2.warpPerspective(pe_filt, M, (frame.shape[1], frame.shape[0]))
            mask = filtOut[:,:,3] != 0

            frame[mask] = np.array([0, 0, 0])
            frame += filtOut[:,:,:3]

            for i in range(4):
                cv2.circle(frame, (int(persp_pts[i,0]), int(persp_pts[i,1])), 5, (0, 0, 255))


        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()