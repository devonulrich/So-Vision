import cv2
import math
import random
import matplotlib.pyplot as plt
import pickle

from mtcnn import MTCNN

FOLDS_PATH = '../FDDB-folds'
FDDB_PATH = '../fddb'
ANIME_PATH = '../cartoon_2000'

TEST_SET_SIZE = 500

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

    intArea = (inthi_x - intlo_x) * (inthi_y - intlo_y)
    if intArea <= 0:
        return 0
    
    area1 = (hi1_x - lo1_x) * (hi1_y - lo1_y)
    area2 = (hi2_x - lo2_x) * (hi2_y - lo2_y)
    unionArea = area1 + area2 - intArea

    return intArea / unionArea

class FDDBImg:
    def __init__(self, path : str, faces : 'list[str]'):
        self.path = path
        self.ref_bboxes = []
        self.pred_bboxes = []
        self.pred_conf = []
        for f in faces:
            self.ref_bboxes.append(self._get_bbox(f))

    # convert FDDB ellipse to a bounding box
    def _get_bbox(self, ellipse : str):
        numStrs = ellipse.split()
        vals = [float(ns) for ns in numStrs]
        # from FDDB readme: <major_axis_radius minor_axis_radius angle center_x center_y detection_score>

        rx = vals[1] * math.cos(math.radians(vals[2]))
        ry = vals[0] * math.cos(math.radians(vals[2]))

        center_x = vals[3]
        center_y = vals[4]

        return (int(center_x - rx), int(center_y - ry), int(rx * 2), int(ry * 2))

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

class ANIMOO:
    def __init__(self, path : str, faces):
        self.path = path
        self.ref_bboxes = []
        self.pred_bboxes = []
        self.pred_conf = []
        for f in faces:
            self.ref_bboxes.append(self._get_bbox(f))

    # convert FDDB ellipse to a bounding box
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


def mainime():
    print("ADVENTURE TIME")
    detector = MTCNN()
    allFaces = []
    dict = {}
    with open(ANIME_PATH + '/icartoon2000.csv') as f:
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
            dict[name].append(splitline[1:])
    for name in dict:
        obj = ANIMOO(ANIME_PATH + '/' + name , dict[name])
        allFaces.append(obj)

    random.shuffle(allFaces)
    testSet = allFaces[:TEST_SET_SIZE] 

    i = 0
    for img in testSet:
        print(i)
        i += 1
        cnnOut = detector.detect_faces(img.get_img())
        for pred_f in cnnOut:
            pred_box = pred_f['box']
            bbox = (pred_box[0], pred_box[1], pred_box[2], pred_box[3])
            img.add_pred(bbox, pred_f['confidence'])
        
        # print('true/false pos/false neg', img.compute_metrics())
        # print(img.pred_conf)
        # img.draw()
        

    with open('anime_mtcnn.pkl', 'wb') as pfile:
        pickle.dump(testSet, pfile)
        


def main():
    print("lame.")
    detector = MTCNN()
    allFaces = []
    with open(FOLDS_PATH + '/allEllipses.txt') as f:
        while True:
            name = f.readline()
            if name == '':
                break
            else:
                name = name[:-1]

            cnt = int(f.readline())
            faces = [f.readline() for i in range(cnt)]

            obj = FDDBImg(FDDB_PATH + '/' + name + '.jpg', faces)
            allFaces.append(obj)
    
    random.shuffle(allFaces)
    testSet = allFaces[:TEST_SET_SIZE] 

    i = 0
    for img in testSet:
        print(i)
        i += 1
        cnnOut = detector.detect_faces(img.get_img())
        for pred_f in cnnOut:
            pred_box = pred_f['box']
            bbox = (pred_box[0], pred_box[1], pred_box[2], pred_box[3])
            img.add_pred(bbox, pred_f['confidence'])
        
        # print('true/false pos/false neg', img.compute_metrics())
        # print(img.pred_conf)
        # img.draw()

    with open('fddb_mtcnn.pkl', 'wb') as pfile:
        pickle.dump(testSet, pfile)
        

if __name__ == '__main__':
    mainime()
    main()