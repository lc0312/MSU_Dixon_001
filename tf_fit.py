# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:11:01 2021

@author: Tech
"""
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


data_dir = pathlib.Path(r'C:\Users\Tech\Desktop\Chao Liu Nanowire\Code_Nanowire\location_Image')

image_train= tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=4,
    image_size=(720,1280),
    seed=128,
    validation_split=0.15,
    subset='training',
    )

image_validation= tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=4,
    image_size=(720,1280),
    seed=128,
    validation_split=0.15,
    subset='validation',
    )

model = tf.keras.Sequential()
model.add (tf.keras.Input (shape=(720,1280,1)))
model.add (tf.keras.layers.Conv2D(256, (2, 2), strides=(1, 1), activation="relu"))
model.add (tf.keras.layers.BatchNormalization())
model.add (tf.keras.layers.MaxPooling2D())
#model.add (tf.keras.layers.)       
model.add (tf.keras.layers.GRU (128, return_sequences=True))
model.add(tf.keras.layers.BatchNormalization())
model.add (tf.keras.layers.GRU (64))
model.add(tf.keras.layers.BatchNormalization())
model.add (tf.keras.layers.Dense(3))

loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
opt=tf.keras.optimizers.Adam(learning_rate=0.0025)

model.compile (loss=loss, optimizer=opt, metrics=["accuracy"])
model.fit (image_train, batch_size=32, epochs=3, verbose=2)
model.evaluate(image_validation, batch_size=32, verbose=2)


