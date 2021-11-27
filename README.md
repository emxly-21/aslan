# ASLAN: American Sign LAnguage Network
A convolutional neural network model to recognize the letters of the American Sign Language alphabet


## Downloading the datasets

### Sign Language MNIST
- Head over to this [link](https://www.kaggle.com/datamunge/sign-language-mnist/download) with a valid Kaggle account to download the dataset. 
- Unzip the downloaded `archive.zip`, rename the directory to `data/`, and move it to the project root directory


### ASL Finger Spelling Dataset
- Heads up! The dataset is fairly large (2.25 GB)
- Run `wget https://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2` in the project root directory
- Extract the file by running `tar xvjf fingerspelling5.tar.bz2`


## Preprocessing
- `get_data` can be used to retrieve the Sign Language MNIST dataset
- `get_data_2` can be used to retrieve the ASL Finger Spelling dataset

