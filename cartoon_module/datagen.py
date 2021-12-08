import numpy as np
import cv2
import pandas

from util import load_img, IoU
from tensorflow.keras.utils import Sequence

class CartoonDataGenerator(Sequence):
    def __init__(self, path, batch_size=128):
        self.path = path
        self.bbox_dict = self._assemble_dict()
        self.files = list(self.bbox_dict.keys())
        self.nfiles = len(self.files)
        self.batch_size = batch_size

    def _assemble_dict(self):
        df = pandas.read_csv(self.path + '/icartoonface_dettrain.csv', 
            names=['file', 'x1', 'y1', 'x2', 'y2'])
        get_bbox = lambda row : (row['x1'], row['y1'], row['x2'] - row['x1'], row['y2'] - row['y1'])

        bbox_dict = {}
        for _, row in df.iterrows():
            fname = row['file']
            if fname in bbox_dict:
                bbox_dict[fname].append(get_bbox(row))
            else:
                bbox_dict[fname] = [get_bbox(row)]
        return bbox_dict

    # number of batches per epoch
    def __len__(self):
        # we take 4 samples per image
        return self.nfiles * 4 // self.batch_size

    # return batch of sample images
    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size // 4 : (idx + 1) * self.batch_size // 4]
        X = []
        y = []
        for file in batch_files:
            img = load_img(self.path + '/icartoonface_dettrain/' + file)
            im1, label1 = self._get_pos(file, img)
            im2, label2 = self._get_neg(file, img)
            im3, label3 = self._get_neg(file, img)
            im4, label4 = self._get_neg(file, img)
            X += [im1, im2, im3, im4]
            y += [label1, label2, label3, label4]
        return np.array(X), np.array(y)

    # get a positive example from a specific file
    # returns ndarray, True/False for image example
    def _get_pos(self, file, img):
        H = img.shape[0]
        W = img.shape[1]
        bboxes = self.bbox_dict[file]
        face_idx = np.random.choice(len(bboxes))
        face_bbox = bboxes[face_idx]
        
        our_size = max(face_bbox[2], face_bbox[3])
        our_size = min(W - face_bbox[0], our_size)
        our_size = min(H - face_bbox[1], our_size)

        our_bbox = (face_bbox[0], face_bbox[1], our_size, our_size)
        
        cropped_img = img[our_bbox[1] : our_bbox[1] + our_bbox[3], our_bbox[0] : our_bbox[0] + our_bbox[2], :]
        return cv2.resize(cropped_img, (224, 224)), True

    # get a negative example from a specific file
    def _get_neg(self, file, img):
        H = img.shape[0]
        W = img.shape[1]
        bboxes = self.bbox_dict[file]
        while True:
            x = np.random.randint(W - 15)
            y = np.random.randint(H - 15)
            sz = np.random.randint(15, 350)
            sz = min(W - x, sz)
            sz = min(H - y, sz)
            our_bbox = (x, y, sz, sz)
            is_face = False
            for face in bboxes:
                if IoU(our_bbox, face) > 0.5:
                    is_face = True
                    break
            if not is_face:
                break

        cropped_img = img[our_bbox[1] : our_bbox[1] + our_bbox[3], our_bbox[0] : our_bbox[0] + our_bbox[2], :]
        return cv2.resize(cropped_img, (224, 224)), False