from mtcnn import MTCNN
import cv2
import sys
import numpy as np
from tensorflow import keras

sys.path.append("../classifier")
sys.path.append("../cartoon_module")

from classify_image import classify_image
from detect import detect_on_img

mtcnn = None
mtcnn_loaded = False

so_vision_model = None
so_vision_model_loaded = False

def mtcnn_detect_faces(image):
    global mtcnn, mtcnn_loaded
    if not mtcnn_loaded:
        mtcnn = MTCNN(steps_threshold=[0.5, 0.0, 0.0])
        mtcnn_loaded = True

    res = []
    mtcnn_output = mtcnn.detect_faces(image)

    for curr in mtcnn_output:
        res.append((curr["box"], curr["confidence"]))

    return res

def so_vision_detect_faces(image):
    global so_vision_model, so_vision_model_loaded
    if not so_vision_model_loaded:
        so_vision_model = keras.models.load_model('../cartoon_module/newmodel15')
        so_vision_model_loaded = True

    res = []
    
    outboxes, outscores = detect_on_img(so_vision_model, image, soft=True)
    for boxIdx in range(len(outscores)):
        pred_box = np.int32(outboxes[boxIdx,:])
        bbox = (pred_box[0], pred_box[1], pred_box[2], pred_box[3])
        res.append((bbox, outscores[boxIdx]))

    return res

def simple_combined_detect_faces(image):
    predictions = mtcnn_detect_faces(image)
    predictions.extend(so_vision_detect_faces(image))
    
    return predictions

def classifier_combined_detect_faces(image):
    cartoon_or_real_class = classify_image([image])

    if cartoon_or_real_class == 0:
        return so_vision_detect_faces(image)
    else:
        return mtcnn_detect_faces(image)

# simple_combined_detect_faces(cv2.cvtColor(cv2.imread("../dataset_generation/so_vision_dataset/so_vision_test_set/img_1001.jpg"), cv2.COLOR_BGR2RGB))


    
    