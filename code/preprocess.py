
from csv import reader
import numpy as np



## 28x28 pixel images
## 27455 samples

## returns 


#  def get_csv_length(file_path): 
#      with open(file_path, 'r') as read_obj:

#          csv_data = reader(read_obj)
#          print(len(list(csv_data)))
#      return

#  get_csv_length('../data/sign_mnist_train.csv')
#  get_csv_length('../data/sign_mnist_test.csv')

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

