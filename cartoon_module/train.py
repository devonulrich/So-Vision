import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import pickle

from datagen import CartoonDataGenerator, assemble_dict

ANIME_PATH = '/scratch/network/dulrich/personai_icartoonface_dettrain'
TRAIN_FRAC = 0.8

def train():
    base_model = keras.applications.VGG16(include_top=False, weights='imagenet', 
        input_shape=(224, 224, 3))
    base_model.summary()
    base_model.trainable = False
    model_input = keras.Input(shape=(224, 224, 3))
    x = tf.cast(model_input, tf.float32)
    x = keras.applications.vgg16.preprocess_input(x)
    
    x = base_model(model_input, training=False)
    # x = keras.layers.Convolution2D(1, 2, 2, activation='relu', name='newconv_1')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(model_input, x)
    crossentropy = keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=keras.optimizers.Adam(), loss=crossentropy, metrics=['binary_accuracy'])
    model.summary()

    all_data = assemble_dict(ANIME_PATH)
    all_files = list(all_data.keys())
    random.shuffle(all_files)

    train_cutoff = int(len(all_files) * TRAIN_FRAC)
    train_gen = CartoonDataGenerator(ANIME_PATH, all_data, all_files[ : train_cutoff])
    val_gen = CartoonDataGenerator(ANIME_PATH, all_data, all_files[train_cutoff : ])

    history = model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)
    model.save('./modelout')
    pickle.dump(history, './history.pkl')

if __name__ == '__main__':
    train()
