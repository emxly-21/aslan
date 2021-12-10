import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Softmax, Flatten, Reshape, BatchNormalization

tf.keras.backend.set_floatx('float64')

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 100
        self.lrdecay = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=10000,
            decay_rate=0.95)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lrdecay)
        self.model = tf.keras.models.load_model('../model_100/')

        # ScienceDirect Implementation
        
        '''self.model.build((28, 28))
        self.model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
        self.model.add(Conv2D(8, 19, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(3,3)))
        self.model.add(Conv2D(16, 17, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(3,3)))
        self.model.add(Conv2D(32, 15, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(3,3)))
        self.model.add(Flatten())
        self.model.add(Softmax())
        self.model.add(Dense(26, activation='relu'))'''
        

        # TowardsDataScience Implementation
        '''self.model.add(Reshape((28, 28, 1)))

        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(2, 2))

        self.model.add(Conv2D(128, (3, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(2, 2))

        self.model.add(Flatten())
        self.model.add(BatchNormalization())
        self.model.add(Dense(256, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dense(26, activation="softmax"))'''


    def call(self, inputs):
        return self.model(inputs)


    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))