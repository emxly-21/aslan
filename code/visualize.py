from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import math

from preprocess import get_data, get_data_2
from constants import LETTERS

import matplotlib as mpl
mpl.use('tkagg')

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
    print(num_images)
    predicted_labels = np.argmax(probabilities, axis=1)

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


aslan_kaggle_model = tf.keras.models.load_model('../model_kaggle/')
kaggle_input, kaggle_labels = get_data("../data/sign_mnist_test.csv")
kaggle_probs = aslan_kaggle_model.predict(kaggle_input)

visualize_results(kaggle_input[:10], kaggle_probs[:10], kaggle_labels[:10])

# aslan_model = tf.keras.models.load_model('../model/')
# model_input, model_labels = get_data_2()
# model_probs = []
# for i in range(10):
#     model_probs.append(aslan_model.predict(model_input[i]))

# visualize_results(model_input[:10], model_probs, model_labels[:10])