from train_model import get_center_slice
import keras
import cv2
import numpy as np

model_loaded = False
model = None
model_name = "50_epoch_model2.h5"

def classify_image(images):
    global model_loaded
    global model
    if not model_loaded:
        print("loading model again")
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

# for x in range(0,2000):
#     input_images.append(cv2.cvtColor(cv2.imread("../dataset_generation/so_vision_dataset/so_vision_test_set/img_" + str(x) + ".jpg"), cv2.COLOR_BGR2RGB))
# input_images.append(cv2.cvtColor(cv2.imread("../test_face_1.jpg"), cv2.COLOR_BGR2RGB))

# res = classify_image(input_images)

# corrects = 0
# for x in range(0,1000):
#     if res[x] == 0:
#         corrects += 1

# for x in range(1000,2000):
#     if res[x] == 1:
#         corrects += 1

# print(corrects/2000)