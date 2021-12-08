# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:11:01 2021

@author: Tech
"""
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image')
validation_dir=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_image_validation')

image_train= tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    #class_names=['left', 'no', 'right'],
    color_mode='grayscale',
    batch_size=5,
    image_size=(720,1280),
    seed=128,
    validation_split=0.1,
    subset='training',
    )

image_validation= tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels='inferred',
    label_mode='int',
    #class_names=['left', 'no', 'right'],
    color_mode='grayscale',
    batch_size=5,
    image_size=(720,1280),
    seed=128,
    validation_split=0.9,
    subset='validation',
    )


'''
datagen = ImageDataGenerator(rescale=1.0/255)
image_train = datagen.flow_from_directory(
    data_dir,
    target_size=(720,1280),
    batch_size=6,
    color_mode="grayscale",
    class_mode="sparse",
    shuffle=True,
    subset="training",
    seed=123,
)

vali_datagen = ImageDataGenerator(rescale=1.0/255)

image_validation = vali_datagen.flow_from_directory(
    validation_dir,
    target_size=(720,1280),
    batch_size=6,
    color_mode="grayscale",
    class_mode="sparse",
    shuffle=False,
    subset="validation",
    seed=123,
)
'''

model = tf.keras.Sequential()
model.add (tf.keras.Input (shape=(720,1280)))
#model.add (tf.keras.layers.Reshape ((720,1280)))
model.add (tf.keras.layers.BatchNormalization())      
model.add (tf.keras.layers.Bidirectional (tf.keras.layers.GRU (128, return_sequences=True)))
model.add (tf.keras.layers.BatchNormalization())
model.add (tf.keras.layers.GRU (64))
model.add (tf.keras.layers.Dense(16,activation='relu'))
model.add (tf.keras.layers.Dense(3,activation='relu'))

loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt=tf.keras.optimizers.Adam(learning_rate=0.00085)

model.compile (loss=loss, optimizer=opt, metrics=["accuracy"])
model.summary ()

model.fit (image_train, batch_size=16, epochs=5, verbose=2)
model.evaluate(image_validation, batch_size=16, verbose=2)

img_0_dir0=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\0\WIN_20211206_093225.JPG')
img_0_dir1=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\0\WIN_20211206_093204.JPG')
img_0_dir2=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\0\WIN_20211206_093235.JPG')
img_0_dir3=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\0\WIN_20211206_093205.JPG')
img_0_dir4=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\0\WIN_20211206_093234.JPG')

img_1_dir0=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\1\WIN_20211206_092438.JPG')
img_1_dir1=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\1\WIN_20211206_092525.JPG')
img_1_dir2=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\1\WIN_20211206_092527.JPG')
img_1_dir3=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\1\WIN_20211206_092516.JPG')
img_1_dir4=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\1\WIN_20211206_092531.JPG')

img_m1_dir0=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\2\WIN_20211206_093100.JPG')
img_m1_dir1=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\2\WIN_20211206_093009.JPG')
img_m1_dir2=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\2\WIN_20211206_093103.JPG')
img_m1_dir3=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\2\WIN_20211206_093025.JPG')
img_m1_dir4=pathlib.Path(r'C:\Users\dixon\Desktop\Nanowire_Code\location_Image\2\WIN_20211206_093054.JPG')

for i in (img_0_dir0,img_0_dir1,img_0_dir2,img_0_dir3,img_0_dir4,
          img_1_dir0,img_1_dir1,img_1_dir2,img_1_dir3,img_1_dir4,
          img_m1_dir0,img_m1_dir1,img_m1_dir2,img_m1_dir3,img_m1_dir4):
    
    image = tf.keras.preprocessing.image.load_img(i,grayscale=True)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    answer=np.argmax(predictions,axis=1)
    print (answer)
    
img_anyone = pathlib.Path(r'C:\Users\dixon\Downloads\WIN_20211207_141714.JPG')
image = tf.keras.preprocessing.image.load_img(img_anyone,grayscale=True)
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])
predictions = model.predict(input_arr)
answer=np.argmax(predictions,axis=1)
print ('try', answer)

