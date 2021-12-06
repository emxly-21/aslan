from matplotlib import pyplot as plt
import os
import tensorflow as tf
import numpy as np
import random
import math
from model import Model
from preprocess import get_data, get_data_2

def train(model, train_inputs, train_labels):
    indices = tf.random.shuffle(np.arange(len(train_inputs)))
    shuffled_inputs = tf.gather(train_inputs, indices)
    shuffled_labels = tf.gather(train_labels, indices)
    accuracy = 0
    batch = 0
    losses = []
    while batch < len(shuffled_inputs):
        flipped = shuffled_inputs[batch:batch + model.batch_size]
        one_hot_labels = tf.one_hot(tf.cast(shuffled_labels[batch:batch + model.batch_size], tf.uint8), 26, axis=1)
        with tf.GradientTape() as tape:
            predictions = model.call(flipped)
            loss = model.loss(predictions, one_hot_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        accuracy += model.accuracy(predictions, one_hot_labels)
        batch += model.batch_size

        print(f'Loss: {str(loss.numpy())}', end='\r')
        losses.append(loss)
    return losses


def test(model, test_inputs, test_labels):
    accuracy = 0
    batch = 0
    while batch < len(test_inputs):
        logits = model.call(test_inputs[batch:batch+model.batch_size])
        one_hot_labels = tf.one_hot(tf.cast(test_labels[batch:batch + model.batch_size], tf.uint8), 26, axis=1)
        accuracy += model.accuracy(logits, one_hot_labels)
        batch += model.batch_size
    return accuracy / len(test_inputs) * model.batch_size

def visualize_loss(losses): 
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  

if __name__ == '__main__':
    model = Model()

    train_inputs, train_labels = get_data("../data/sign_mnist_train.csv")
    test_inputs, test_labels = get_data("../data/sign_mnist_test.csv")

    losses = []
    for epoch in range(20):
        print(f'Epoch {epoch + 1}:')
        out = train(model, train_inputs, train_labels)
        losses += out
        print(f'\nTest Accuracy: {test(model, test_inputs, test_labels).numpy()}')

    visualize_loss(losses)
    print(f'Test Accuracy: {test(model, test_inputs, test_labels).numpy()}')

    model.save('../model/')