import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Dense, Conv2D, MaxPooling2D, Softmax

class CNN(tf.keras.Model):
    def __init__(self):
        self.batch_size = 100
        self.learning_rate = 5e-3
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model = Sequential()

        model.add(Conv2D(8, 19, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(3,3)))
        model.add(Conv2D(16, 17, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(3,3)))
        model.add(Conv2D(32, 15, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(3,3)))
        model.add(Softmax())
        model.add(Dense(24))


    def call(self, inputs):
        return model(inputs)


    def accuracy_function(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


    def loss_function(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))