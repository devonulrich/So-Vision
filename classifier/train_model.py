import numpy as np
import os
import pickle

import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

image_slice_size = 160
image_slice_shape = (image_slice_size,image_slice_size,3)

dataset_size = 25000
testset_proportion = .10

def get_center_slice(image):
    if image.shape[0] < image_slice_size or image.shape[1] < image_slice_size:
        return None, False
    x_lower = (image.shape[0] - image_slice_size) // 2
    y_lower = (image.shape[1] - image_slice_size) // 2
    return image[x_lower:x_lower + image_slice_size, y_lower:y_lower + image_slice_size, :], True

def get_jpgs_in_directory(path):
    res = []
    for root, dirs, files in os.walk(path):
        for curr_dir in dirs:
            res.extend(get_jpgs_in_directory(curr_dir))
        for file_name in files:
            if file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg"):
                res.append(os.path.join(root, file_name).replace("\\", "/"))
    return res

def process_images(dataset_folder):
    image_paths = get_jpgs_in_directory(dataset_folder)

    # for curr_descriptor in descriptor_paths:
    #     image_info.extend(list(np.genfromtxt(curr_descriptor, delimiter="\n", dtype=str, encoding="utf-8")))
    if dataset_size > len(image_paths):
        exit()

    input_indices = set()

    center_slices = []

    while len(center_slices) < dataset_size:
        row = None

        while not row or row in input_indices:
            row = np.random.randint(len(image_paths))
        input_indices.add(row)
        
        path = image_paths[row]

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        center_slice, worked = get_center_slice(img)
        if worked:
            center_slices.append(center_slice)

    return center_slices

def main():
    np.random.seed(57)
    # Process carton images
    cartoon_images = None
    cartoon_save_file = "cartoon_images_" + str(image_slice_size) + ".pkl"
    if os.path.exists(cartoon_save_file):
        with open(cartoon_save_file, 'rb') as f:
            cartoon_images = pickle.load(f)
    else:
        cartoon_images = process_images("../personai_icartoonface_rectrain/")
        with open(cartoon_save_file, 'wb') as f:
            pickle.dump(cartoon_images, f)

    train_end_index = int((1 - testset_proportion) * len(cartoon_images))

    test_cartoon = cartoon_images[train_end_index:]
    cartoon_images = cartoon_images[0:train_end_index]

    # Process real images
    regular_images = None
    regular_save_file = "regular_images_" + str(image_slice_size) + ".pkl"
    if os.path.exists(regular_save_file):
        with open(regular_save_file, 'rb') as f:
            regular_images = pickle.load(f)
    else:
        regular_images = process_images("../fddb")
        with open(regular_save_file, 'wb') as f:
            pickle.dump(regular_images, f)

    train_end_index = int((1 - testset_proportion) * len(regular_images))

    test_regular = regular_images[train_end_index:]
    regular_images = regular_images[0:train_end_index]

    # Setup training data
    data_X = cartoon_images.copy()
    data_X.extend(regular_images)

    data_X = np.array(data_X)

    # Class 0 is cartoon, class 1 is regular
    data_Y = np.zeros((len(cartoon_images) + len(regular_images), 2))
    data_Y[0:len(cartoon_images), 0] = 1
    data_Y[len(cartoon_images):len(cartoon_images) + len(regular_images), 1] = 1

    # Setup test data
    test_X = test_cartoon.copy()
    test_X.extend(test_regular)
    test_X = np.array(test_X)
    
    test_Y = np.zeros((len(test_cartoon) + len(test_regular), 2))
    test_Y[0:len(test_cartoon), 0] = 1
    test_Y[len(test_cartoon):len(test_cartoon) + len(test_regular), 1] = 1
    
    #Create the model
    model = Sequential()

    # Convolutional Layer(s)
    model.add(Conv2D(15, 3, data_format="channels_last", activation="relu", input_shape=image_slice_shape))
    # 2nd conv layer
    model.add(Conv2D(15, 3, data_format="channels_last", activation="relu", input_shape=image_slice_shape))
    
    # Pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding="same", data_format="channels_last"))

    model.add(Conv2D(15, 3, data_format="channels_last", activation="relu", input_shape=image_slice_shape))
    model.add(Conv2D(15, 3, data_format="channels_last", activation="relu", input_shape=image_slice_shape))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding="same", data_format="channels_last"))

    # Flatten
    model.add(Flatten())

    #fully connected layer
    # model.add(Dense(((image_slice_size - 2) / 2) ** 2 * 3, activation="softmax"))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()

    model.fit(data_X, data_Y, validation_data=(test_X, test_Y), batch_size=512, epochs=100)

    # 128_slice_15000_double_conv_maxpool = 7 conv 7 conv pool dense
    # 128_slice_25000_double_conv_maxpool = 4 conv 4 conv pool dense
    # 20000_doubleconv_maxpool = 3 conv 3 conv pool dense
    # 128_slice_25000_doubleconv5_maxpool = 5 conv 5 conv pool dense
    # 128_slice_25000_doubleconv5 = 5 conv 5 conv dense
    # 160_slice_25000_tripleconv5_pool = 5 conv 5 conv pool 5 conv dense 88% test accuracy
    # 160_slice_25000_quadconv15_2pool = 15 conv 15 conv pool 15 conv 15conv pool dense
    model.save("saved_models/160_slice_25000_quadconv15_2pool.h5")

if __name__ == "__main__":
    main()