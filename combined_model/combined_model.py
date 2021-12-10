from mtcnn import MTCNN
import cv2

mtcnn = None
mtcnn_loaded = False

so_vision_model = None
so_vision_model_loaded = False

def detect_faces(image):
    global mtcnn, mtcnn_loaded, so_vision_model, so_vision_model_loaded
    if not mtcnn_loaded:
        mtcnn = MTCNN()
        mtcnn_loaded = True
    if not so_vision_model_loaded:
        # load our model
        so_vision_model_loaded = True

    predictions = []

    mtcnn_output = mtcnn.detect_faces(image)

    for curr in mtcnn_output:
        predictions.append((curr["box"], curr["confidence"]))

    output = so_vision_model.predict(image)

    for curr in output:
        predictions.append(  ......   )

    return predictions

detect_faces(cv2.cvtColor(cv2.imread("../dataset_generation/so_vision_dataset/so_vision_test_set/img_1001.jpg"), cv2.COLOR_BGR2RGB))


    
    