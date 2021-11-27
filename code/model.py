import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Softmax, Flatten, Reshape

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 100
        self.learning_rate = 5e-3
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model = Sequential()

        self.model.add(Reshape((28, 28, 1)))
        self.model.add(Conv2D(8, 19, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(3,3)))
        self.model.add(Conv2D(16, 17, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(3,3)))
        self.model.add(Conv2D(32, 15, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(3,3)))
        self.model.add(Flatten())
        self.model.add(Softmax())
        self.model.add(Dense(24, activation='relu'))


    def call(self, inputs):
        return self.model(inputs)


    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))