from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import math

from preprocess import get_data, get_data_2, get_data_3
from constants import LETTERS

import matplotlib as mpl
mpl.use('tkagg')

def triple_visualize_results(image_inputs, prob1, prob2, prob3, image_labels):
    images = np.reshape(image_inputs, (-1, 28, 28))
    num_images = images.shape[0]
    labels1 = np.argmax(np.asarray(prob1).reshape(10, 26), axis=1)
    labels2 = np.argmax(np.asarray(prob2).reshape(10, 26), axis=1)
    labels3 = np.argmax(np.asarray(prob3).reshape(10, 26), axis=1)

    predicted_letters1 = []
    predicted_letters2 = []
    predicted_letters3 = []
    image_letters = []

    for i in range(num_images):
        predicted_letters1.append(LETTERS[labels1[i]+1].upper())
        predicted_letters2.append(LETTERS[labels2[i]+1].upper())
        predicted_letters3.append(LETTERS[labels3[i]+1].upper())
        image_letters.append(LETTERS[int(image_labels[i])+1].upper())

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="MNIST: {}\nFS: {}\nCombo: {}\nActual: {}".format(predicted_letters1[ind], predicted_letters2[ind], predicted_letters3[ind], image_letters[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up 
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    num_images = images.shape[0]
    predicted_labels = np.argmax(np.asarray(probabilities).reshape(10, 26), axis=1)

    predicted_letters = []
    image_letters = []

    for i in range(num_images):
        predicted_letters.append(LETTERS[predicted_labels[i]+1].upper())
        image_letters.append(LETTERS[int(image_labels[i])+1].upper())

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="Predicted: {}\nActual: {}".format(predicted_letters[ind], image_letters[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

# USING SIGN LANGUAGE MNIST DATASET
# aslan_kaggle_model = tf.keras.models.load_model('../model_kaggle/')
# kaggle_input, kaggle_labels = get_data("../data/sign_mnist_test.csv")
# kaggle_probs = aslan_kaggle_model.predict(kaggle_input)

# visualize_results(kaggle_input[:10], kaggle_probs[:10], kaggle_labels[:10])


# USING FINGERSPELLING A DATASET
# aslan_model = tf.keras.models.load_model('../model/')
# model_input, model_labels = get_data_2()
# indices = tf.random.shuffle(np.arange(len(model_input)))
# shuffled_inputs = tf.gather(model_input, indices)
# shuffled_labels = tf.gather(model_labels, indices)
# model_probs = aslan_model.predict(np.reshape(shuffled_inputs[:10], (-1, 28, 28)))

# visualize_results(shuffled_inputs[:10], model_probs, shuffled_labels[:10])


# USING CUSTOM DATASET
aslan_kaggle_model = tf.keras.models.load_model('../model_kaggle/')
aslan_model = tf.keras.models.load_model('../model/')
aslan_combo_model = tf.keras.models.load_model('../model_100/')

model_input, model_labels = get_data_3()

indices = tf.random.shuffle(np.arange(len(model_input)))
shuffled_inputs = tf.gather(model_input, indices)
shuffled_labels = tf.gather(model_labels, indices)

kaggle_probs = aslan_kaggle_model.predict(np.reshape(shuffled_inputs[:10], (-1, 784)))
model_probs = aslan_model.predict(np.reshape(shuffled_inputs[:10], (-1, 28, 28)))
combo_probs = aslan_combo_model.predict(np.reshape(shuffled_inputs[:10], (-1, 28, 28)))

triple_visualize_results(shuffled_inputs[:10], kaggle_probs, model_probs, combo_probs, shuffled_labels[:10])