
from csv import reader
import numpy as np


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

    with open(file_path, 'r') as file:

        csv_data = reader(file)

        num_samples = len(list(csv_data))
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

    return inputsVector, labelsVector


#  inputsVector, labelsVector = get_data("../data/sign_mnist_train.csv")

#  print(inputsVector)
#  print(labelsVector)
#  print("inputs shape: ", inputsVector.shape)
#  print("labels shape: ", labelsVector.shape)

