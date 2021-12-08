import numpy as np
import tensorflow as tf
from tensorflow import keras

from datagen import CartoonDataGenerator

ANIME_PATH = '/Users/devon/Desktop/personai_icartoonface_dettrain'

def train():
    base_model = keras.applications.VGG16(include_top=False, weights='imagenet')
    base_model.trainable = False
    model_input = keras.Input(shape=(224, 224, 3))
    x = base_model(model_input, training=False)
    x = keras.layers.Convolution2D(1, 2, 2, activation='relu', name='newconv_1')(x)
    x = keras.layers.Flatten()(x)


    model = keras.Model(model_input, x)
    model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')

    gen = CartoonDataGenerator(ANIME_PATH)
    model.fit(gen, verbose=1, use_multiprocessing=True)

if __name__ == '__main__':
    train()