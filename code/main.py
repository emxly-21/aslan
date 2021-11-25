from matplotlib import pyplot as plt
import os
import tensorflow as tf
import numpy as np
import random
import math
from model import CNN

def train(model, train_inputs, train_labels):
    indices = tf.random.shuffle(np.arange(len(train_inputs)))
    shuffled_inputs = tf.gather(train_inputs, indices)
    shuffled_labels = tf.gather(train_labels, indices)
    accuracy = 0
    batch = 0
    while batch < len(shuffled_inputs):
        flipped = tf.image.random_flip_left_right(shuffled_inputs[batch:batch + model.batch_size])
        with tf.GradientTape() as tape:
            predictions = model.call(flipped)
            loss = model.loss(predictions, shuffled_labels[batch:batch + model.batch_size])
        model.loss_list.append(loss.numpy())
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        accuracy += model.accuracy(predictions, shuffled_labels[batch:batch + model.batch_size])
        batch += model.batch_size
    print('Accuracy: ' + str(accuracy.numpy() / len(shuffled_inputs) * model.batch_size))


def test(model, test_inputs, test_labels):
    accuracy = 0
    batch = 0
    while batch < len(test_inputs):
        logits = model.call(test_inputs[batch:batch+model.batch_size], is_testing=True)
        accuracy += model.accuracy(logits, test_labels[batch:batch+model.batch_size])
        batch += model.batch_size
    return accuracy / len(test_inputs) * model.batch_size

if __name__ == '__main__':
    model = CNN()
    train(model, _, _)
    test(model, _, _)