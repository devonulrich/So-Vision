import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import pickle
import cv2
import matplotlib.pyplot as plt

import detect
import util

ICARTOON_PATH = '/scratch/network/dulrich/personai_icartoonface_dettrain'

TEST_SET_SIZE = 100

SOFT = False

class CartoonImg:
    def __init__(self, path : str, faces):
        self.path = path
        self.ref_bboxes = []
        self.pred_bboxes = []
        self.pred_conf = []
        for f in faces:
            self.ref_bboxes.append(self._get_bbox(f))

    # convert strs to a bounding box
    def _get_bbox(self, vals):
        data = [int(val) for val in vals]
        # from cartoon dataset readme x1 y1 x2 y2
        return (data[0], data[1], data[2]-data[0], data[3]-data[1])

    def add_pred(self, bbox, conf):
        self.pred_bboxes.append(bbox)
        self.pred_conf.append(conf)

    def get_img(self):
        # return plt.imread(self.path)
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
                iou = util.IoU(curr_bbox, self.ref_bboxes[j])
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

def main():
    model = keras.models.load_model('./modelout')
    allFaces = []
    dict = {}
    with open(ICARTOON_PATH + '/icartoonface_dettrain.csv') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            else:
                line = line[:-1]
            splitline = line.split(',')
            name = splitline[0]
            if(name not in dict):
                dict[name] = []
            dict[name].append(splitline[1:5])
    for name in dict:
        obj = CartoonImg(ICARTOON_PATH + '/icartoonface_dettrain/' + name , dict[name])
        allFaces.append(obj)

    random.shuffle(allFaces)
    testSet = allFaces[:TEST_SET_SIZE] 

    print('done loading', flush=True)

    i = 0
    for img in testSet:
        print(i, flush=True)
        i += 1
        outboxes, outscores = detect.detect_on_img(model, img.get_img(), soft=SOFT)
        for boxIdx in range(len(outscores)):
            pred_box = outboxes[boxIdx,:]
            bbox = (pred_box[0], pred_box[1], pred_box[2], pred_box[3])
            img.add_pred(bbox, outscores[boxIdx])
        

    suffix = 'soft.pkl' if SOFT else 'reg.pkl'
    outputName = 'cartoon_custom_new_' + suffix
    with open(outputName, 'wb') as pfile:
        pickle.dump(testSet, pfile)
        
if __name__ == '__main__':
    main()
