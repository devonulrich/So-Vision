from mtcnn import MTCNN
import cv2
import sys
import numpy as np
from tensorflow import keras
import pickle

sys.path.append("../classifier")
sys.path.append("../cartoon_module")

from classify_image import classify_image
from detect import detect_on_img
from so_evaluate import SoVisionImg

# sys.path.append("../baseline")
# from evaluate import ANIMOO

mtcnn = None
mtcnn_loaded = False

sv_cartoons = None
sv_real = None
so_vision_model_loaded = False

def mtcnn_detect_faces(image):
    global mtcnn, mtcnn_loaded
    if not mtcnn_loaded:
        mtcnn = MTCNN()
        mtcnn_loaded = True

    mtcnn_output = mtcnn.detect_faces(image.get_img())

    for curr in mtcnn_output:
        image.add_pred(curr["box"], curr["confidence"])

    return image

# def so_vision_detect_faces(image):
#     global so_vision_model, so_vision_model_loaded
#     if not so_vision_model_loaded:
#         so_vision_model = keras.models.load_model('../cartoon_module/newmodel15')
#         so_vision_model_loaded = True

#     res = []

#     outboxes, outscores = detect_on_img(so_vision_model, image, soft=True)
#     for boxIdx in range(len(outscores)):
#         pred_box = np.int32(outboxes[boxIdx,:])
#         bbox = (pred_box[0], pred_box[1], pred_box[2], pred_box[3])
#         res.append((bbox, outscores[boxIdx]))

#     return res

# def mtcnn_detect_faces(idx):
#     global mtcnn, mtcnn_loaded
#     if not mtcnn_loaded:
#         with open('../baseline/so_mtcnn.pkl', 'rb') as pfile:
#             mtcnn = pickle.load(pfile)
#         mtcnn_loaded = True

#     return mtcnn[idx]

def so_vision_detect_faces(idx):
    global sv_cartoons, sv_real, so_vision_model_loaded
    if not so_vision_model_loaded:
        with open('../cartoon_module/so_ourmodel.pkl', 'rb') as pfile:
            sv_cartoons = pickle.load(pfile)
        with open('../cartoon_module/so_ourmodel_fddb.pkl', 'rb') as pfile:
            sv_real = pickle.load(pfile)
        so_vision_model_loaded = True

    return sv_real[idx - 1000] if idx >= 1000 else sv_cartoons[idx]

def union_combined_detect_faces(image, idx):
    mtcnn_img = mtcnn_detect_faces(image)
    sv_img = so_vision_detect_faces(idx)
    
    new_preds = []

    for x, box in enumerate(mtcnn_img.pred_bboxes):
        new_preds.append([tuple(box), mtcnn_img.pred_conf[x], 1])

    for x, box in enumerate(sv_img.pred_bboxes):
        new_preds.append([tuple(box), sv_img.pred_conf[x], 1])

    mtcnn_img.pred_bboxes = []
    mtcnn_img.pred_conf = []

    while True:
        changed = False

        if not new_preds:
            break
        
        new_new_preds = [new_preds[0]]

        for x in range(1, len(new_preds)):
            merged = False
            x1, y1, width, height = new_preds[x][0]
            x2, y2 = x1 + width, y1 + height

            for y in range(len(new_new_preds)):
                px1, py1, pw, ph = new_new_preds[y][0]
                px2, py2 = px1 + pw, py1 + ph

                if not (x1 > px2 or px1 > x2 or y1 > py2 or py1 > y2):
                    new_x1, new_y1 = min(x1, px1), min(y1, py1)
                    new_x2, new_y2 = max(x2, px2), max(y2, py2)
                    new_width, new_height = new_x2 - new_x1, new_y2 - new_y1
                    new_new_preds[y] = [(new_x1, new_y1, new_width, new_height), new_new_preds[y][1] + new_preds[y][1], new_new_preds[y][2] + 1]
                    changed = True
                    merged = True
            if not merged:
                new_new_preds.append(new_preds[x])

        new_preds = new_new_preds
        if not changed:
            break

    for x in range(len(new_preds)):
        mtcnn_img.pred_bboxes.append(new_preds[x][0])
        mtcnn_img.pred_conf.append(new_preds[x][1] / new_preds[x][2])

    return mtcnn_img

def classifier_combined_detect_faces(image, idx):
    cartoon_or_real_class = classify_image([image.get_img()])

    if cartoon_or_real_class[0] == 0:
        return so_vision_detect_faces(idx)
    else:
        return mtcnn_detect_faces(image)

    
    