import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow import keras
from cartoon_module.detect import detect_on_img

SO_VISION_PATH = "../dataset_generation/so_vision_dataset"

# rects are (x, y, width, height)
def IoU(rect1, rect2):
    lo1_x = rect1[0]
    lo1_y = rect1[1]
    hi1_x = rect1[0] + rect1[2]
    hi1_y = rect1[1] + rect1[3]

    lo2_x = rect2[0]
    lo2_y = rect2[1]
    hi2_x = rect2[0] + rect2[2]
    hi2_y = rect2[1] + rect2[3]

    intlo_x = max(lo1_x, lo2_x)
    intlo_y = max(lo1_y, lo2_y)
    inthi_x = min(hi1_x, hi2_x)
    inthi_y = min(hi1_y, hi2_y)

    if inthi_x - intlo_x <= 0 or inthi_y - intlo_y <= 0:
        return 0

    intArea = (inthi_x - intlo_x) * (inthi_y - intlo_y)
    
    area1 = (hi1_x - lo1_x) * (hi1_y - lo1_y)
    area2 = (hi2_x - lo2_x) * (hi2_y - lo2_y)
    unionArea = area1 + area2 - intArea
    if unionArea <= 0:
        return 0

    return intArea / unionArea


class SoVisionImg:
    def __init__(self, path : str, faces):
        self.path = path
        self.ref_bboxes = []
        self.pred_bboxes = []
        self.pred_conf = []
        for f in faces:
            self.ref_bboxes.append(self._get_bbox(f))

    def _get_bbox(self, vals):
        data = [int(val) for val in vals]
        return (data[0], data[1], data[2]-data[0], data[3]-data[1])

    def add_pred(self, bbox, conf):
        self.pred_bboxes.append(bbox)
        self.pred_conf.append(conf)

    def get_img(self):
        return cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)

    def compute_metrics(self, threshold=0.0, returnArrs=False):
        pred_det = [False for b in self.pred_bboxes]
        ref_det = [False for b in self.ref_bboxes]
        false_pos_cnt = 0
        for i in range(len(self.pred_bboxes)):
            curr_bbox = self.pred_bboxes[i]
            if self.pred_conf[i] < threshold:
                continue

            maxIoU = 0
            maxIdx = 0
            for j in range(len(self.ref_bboxes)):
                iou = IoU(curr_bbox, self.ref_bboxes[j])
                if iou > maxIoU:
                    maxIoU = iou
                    maxIdx = j
            
            # compute true pos, false pos, false neg for all pred/ref bboxes
            # true pos: IoU is over .5, maxIdx isn't flagged
            if maxIoU >= 0.5 and not ref_det[maxIdx]:
                ref_det[maxIdx] = True
                pred_det[i] = True
            else:
                # false pos: IoU less than .5 or maxIdx is flagged
                false_pos_cnt += 1

        # false neg: maxIdx isn't flagged after all pred bboxes
        false_neg_cnt = 0
        true_pos_cnt = 0
        for d in ref_det:
            if d:
                true_pos_cnt += 1
            else:
                false_neg_cnt += 1
        
        if returnArrs:
            return (true_pos_cnt, false_pos_cnt, false_neg_cnt, pred_det, ref_det)
        else:
            return (true_pos_cnt, false_pos_cnt, false_neg_cnt)

    
    def draw(self):
        img = plt.imread(self.path)
        for bbox in self.ref_bboxes:
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0))
        for bbox in self.pred_bboxes:
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255))

        plt.imshow(img)
        plt.show()

def read_annotation(f):
    first_line = f.readline()
    name, curr_class = first_line.split(" ")

    cnt = int(f.readline())
    faces = []

    for _ in range(cnt):
        line = f.readline().split(" ")
        x1 = int(line[0])
        y1 = int(line[1])
        width = int(line[2])
        height = int(line[3][:-1])
        faces.append((x1, y1, x1 + width, y1 + height))

    image_path = SO_VISION_PATH + '/' + name

    obj = SoVisionImg(image_path, faces)
    return obj

def get_anime_faces_from_our_set():
    allFaces = []

    with open(SO_VISION_PATH + "/annotations.txt") as f:
        for _ in range(1000):
            allFaces.append(read_annotation(f))

    return allFaces

def get_real_faces_from_our_set():
    allFaces = []

    with open(SO_VISION_PATH + "/annotations.txt") as f:
        for _ in range(1000):
            read_annotation(f)
        for _ in range(1000):
            allFaces.append(read_annotation(f))

    return allFaces

def get_all_faces_from_our_set():
    allFaces = []

    with open(SO_VISION_PATH + "/annotations.txt") as f:
        for _ in range(1000):
            allFaces.append(read_annotation(f))
        for _ in range(1000):
            allFaces.append(read_annotation(f))

    return allFaces

def our_model_eval(model, allFaces):
    i = 0
    for img in allFaces:
        print(i, '/', len(allFaces), flush=True)
        i += 1
        outboxes, outscores = detect_on_img(model, img.get_img(), soft=True)
        for boxIdx in range(len(outscores)):
            pred_box = np.int32(outboxes[boxIdx,:])
            bbox = (pred_box[0], pred_box[1], pred_box[2], pred_box[3])
            img.add_pred(bbox, outscores[boxIdx])

        metrics = img.compute_metrics(returnArrs=True)
        print('Metrics:', metrics[:3], flush=True)
        
    with open('so_ourmodel.pkl', 'wb') as pfile:
        pickle.dump(allFaces, pfile)

if __name__ == '__main__':
    model = keras.models.load_model('./newmodel15')
    faces = get_anime_faces_from_our_set()
    our_model_eval(model, faces)