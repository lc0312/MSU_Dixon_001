import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_batch_ops import batch
import time

with tf.device('/CPU:0'):

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data() # shape(28,28) 60000 sample
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.Sequential ()
    model.add (tf.keras.Input (shape=(28,28)))
    model.add (tf.keras.layers.GRU (128, return_sequences=True))
    model.add (tf.keras.layers.GRU (64))
    model.add (tf.keras.layers.Dense(10))

    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt=tf.keras.optimizers.Adam(learning_rate=0.0025)

    model.compile (loss=loss, optimizer=opt, metrics=["accuracy"])

    start_time=time.time()
    model.fit (x_train, y_train, batch_size=64, epochs=2, verbose=2)
    end_time=time.time()

    model.evaluate(x_test, y_test, batch_size=64, verbose=2)

print (start_time-end_time)

