from numpy.lib.type_check import real
from train_model import get_center_slice
import keras
import cv2
import numpy as np

model_loaded = False
model = None
model_name = "160_slice_25000_quadconv15_2pool.h5"

def inference(images):
    global model_loaded
    global model
    if not model_loaded:
        model = keras.models.load_model("saved_models/" + model_name)
        model_loaded = True

    inputs = []
    works = [False] * len(images)

    for x, image in enumerate(images):
        center_slice, worked = get_center_slice(image)
        
        if not worked:
            continue
        works[x] = True
    
        inputs.append(center_slice)

    inputs = np.array(inputs)
    output = model.predict(inputs)

    real_output = []

    for x in range(len(images)):
        if works[x]:
            real_output.append(np.argmax(output[x]))
        else:
            real_output.append(-1)
    
    return real_output

# input_images = []

# for x in range(0,5):
#     input_images.append(cv2.cvtColor(cv2.imread("../test_face_" + str(x) + ".jpg"), cv2.COLOR_BGR2RGB))
# input_images.append(cv2.cvtColor(cv2.imread("../test_face_1.jpg"), cv2.COLOR_BGR2RGB))

# inference(input_images)