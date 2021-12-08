import numpy as np
import os
import sys
import random
from collections import defaultdict
from shutil import copy
from PIL import Image

# import stuff from evaluate
sys.path.append("../baseline")
from evaluate import ANIMOO, FDDBImg, get_real_faces_from_file

cartoon_set_path = "../personai_icartoonface_dettrain/icartoonface_dettrain"
real_set_path = "../fddb"
new_dataset_annotation_path = "./so_vision_dataset"
new_dataset_image_path = "./so_vision_dataset/so_vision_test_set"

images_per_type = 1000
min_size = 160

# Note: since ANIMOO and FDDBImg both have the ref_bboxes variable, this works, but this is bad practice :) it wouldn't be tho if they inherited from something
def write_image_to_file(curr_file, image, new_path, curr_class):
    curr_file.write(new_path + " " + str(curr_class) + "\n")
    curr_file.write(str(len(image.ref_bboxes)) + "\n")

    for bbox in image.ref_bboxes:
        curr_file.write(str(bbox[0]) + " ")
        curr_file.write(str(bbox[1]) + " ")
        curr_file.write(str(bbox[2]) + " ")
        curr_file.write(str(bbox[3]) + "\n")

def generate_dataset():
    cartoon_images = []
    real_images = []

    for x in range(1,11):
        current_descriptor_path = "../FDDB-folds/FDDB-fold-" + f"{x:02d}" + "-ellipseList.txt"
        real_images.extend(get_real_faces_from_file(current_descriptor_path, min_size))
    
    cartoon_image_descriptors = np.genfromtxt("../personai_icartoonface_dettrain/icartoonface_dettrain.csv", dtype=str, delimiter="\n", encoding="utf-8")
    cartoon_boxes = defaultdict(list)

    for line in cartoon_image_descriptors:
        split_arr = line.split(",")
        path = cartoon_set_path + "/" + split_arr[0]
        bounding_box = split_arr[1:] 
        cartoon_boxes[path].append(bounding_box)

    for path, bounding_boxes in cartoon_boxes.items():
        if os.path.exists(path):
            im = Image.open(path)
            if im.width < min_size or im.height < min_size:
                continue
            cartoon_images.append(ANIMOO(path, bounding_boxes))

    # set seed for reproducibility
    random.seed(123)
    random.shuffle(cartoon_images)
    random.shuffle(real_images)

    dataset_cartoon_images = cartoon_images[:images_per_type]
    dataset_real_images = real_images[:images_per_type]

    with open(new_dataset_annotation_path + "/annotations.txt", "w") as f:
        index = 0
        for image in dataset_cartoon_images:
            new_path_ending = "/img_" + str(index) + ".jpg"
            # the 0 represents that this is a cartoon image
            write_image_to_file(f, image, "so_vision_test_set" + new_path_ending, 0)
            copy(image.path, new_dataset_image_path + new_path_ending)
            index += 1

        for image in dataset_real_images:
            new_path_ending = "/img_" + str(index) + ".jpg"
            # 1 meaning its a real image
            write_image_to_file(f, image, "so_vision_test_set" + new_path_ending, 1)
            copy(image.path, new_dataset_image_path + new_path_ending)
            index += 1

if __name__ == "__main__":
    generate_dataset()