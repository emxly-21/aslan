
from csv import reader
import numpy as np
from PIL import Image
import os

## Sign Language MNIST
def get_data(file_path): 
    """
    Returns two numpy arrays, 

    inputs shape:  (num_samples, 784)
    labels shape:  (num_samples,)

    Training data contains 27455 samples
    Testing data contains 7172 samples

    Example usage: 
    `inputsVector, labelsVector = get_data("../data/sign_mnist_train.csv")`

    Images are 28x28 pixels
    """

    print("reading file_path: ", file_path)
    with open(file_path, 'r') as file:

        csv_data = list(reader(file))

        num_samples = len(csv_data)
        inputsVector = np.empty([num_samples, 784])
        labelsVector = np.empty([num_samples])
        is_header = True

        count = 0
        for row in csv_data:
            if is_header: 
                is_header = False
                continue

            inputsVector[count] = row[1:]
            labelsVector[count] = row[0]
            
            count += 1

    return inputsVector / 255, labelsVector


def image_to_np_array(file_path): 
    image = Image.open(file_path).convert('L') # convert to grayscale
    image = image.resize((28, 28), resample=Image.BICUBIC)
    data = np.asarray(image)
    return data


## ASL Finger Spelling Dataset
def get_data_2(): 
    """
    Returns two numpy arrays, 

    inputs shape:  (num_samples, height, width)
    labels shape:  (num_samples, height, width)

    Training data (A) contains 12547 samples
    Training data (A, B) contains 26445 samples

    Example usage: 
    `inputsVector, labelsVector = get_data()`

    Images have varying widths and heights
    """

    #  datasets = ['A/', 'B/']
    datasets = ['A/']#, 'B/', 'C/', 'D/', 'E/']

    inputsVector = []
    labelsVector = []

    for letter_index in range(26): 
        if letter_index in [9, 25]: # ignore j and z
            continue

        counter = 0
        letter = chr(letter_index + 97)
        letter_dir = letter + "/"
        for a_dataset_dir in datasets: 
            a_file_path = "../dataset5/" + a_dataset_dir + letter_dir
            for a_file in os.listdir(a_file_path): 
                #  image = Image.open(file_path)

                if a_file.startswith("color_"): 
                    #  print(a_file_path + a_file)
                    data = image_to_np_array(a_file_path + a_file)

                    inputsVector.append(data)
                    labelsVector.append(letter_index)

                    #  print(data)
                    #  print(type(data))
                    #  print(data.shape)
                    counter += 1
        print("letter ", letter, " has ", counter)

    inputsVector = np.stack(inputsVector, axis=0).astype(np.float32) / 255
    labelsVector = np.stack(labelsVector, axis=0)
    #  print(inputsVector)
    #  print(labelsVector)

    #  print(type(inputsVector))
    #  print(inputsVector.shape)
    #  print(type(labelsVector))
    #  print(labelsVector.shape)

    return inputsVector, labelsVector


## ASL Finger Spelling Dataset
def get_data_3(): 
    """
    Returns two numpy arrays, 

    inputs shape:  (num_samples, height, width)
    labels shape:  (num_samples, height, width)

    Training data (A) contains 12547 samples
    Training data (A, B) contains 26445 samples

    Example usage: 
    `inputsVector, labelsVector = get_data()`

    Images have varying widths and heights
    """

    #  datasets = ['A/', 'B/']
    datasets = ['A/']#, 'B/', 'C/', 'D/', 'E/']

    inputsVector = []
    labelsVector = []

    for letter_index in range(3): 
        if letter_index in [9, 25]: # ignore j and z
            continue

        counter = 0
        letter = chr(letter_index + 97)
        letter_dir = letter + "/"
        for a_dataset_dir in datasets: 
            a_file_path = "../data/custom_dataset/" + a_dataset_dir + letter_dir
            for a_file in os.listdir(a_file_path): 
                #  image = Image.open(file_path)

                if a_file.startswith("color_"): 
                    #  print(a_file_path + a_file)
                    data = image_to_np_array(a_file_path + a_file)

                    inputsVector.append(data)
                    labelsVector.append(letter_index)

                    #  print(data)
                    #  print(type(data))
                    #  print(data.shape)
                    counter += 1
        print("letter ", letter, " has ", counter)

    inputsVector = np.stack(inputsVector, axis=0).astype(np.float32) / 255
    labelsVector = np.stack(labelsVector, axis=0)
    #  print(inputsVector)
    #  print(labelsVector)

    #  print(type(inputsVector))
    #  print(inputsVector.shape)
    #  print(type(labelsVector))
    #  print(labelsVector.shape)

    return inputsVector, labelsVector


#  get_data_2()

#  inputsVector, labelsVector = get_data("../data/sign_mnist_train.csv")

#  print(inputsVector)
#  print(labelsVector)
#  print("inputs shape: ", inputsVector.shape)
#  print("labels shape: ", labelsVector.shape)


