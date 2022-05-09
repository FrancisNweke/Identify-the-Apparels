import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
from tqdm import tqdm
import numpy as np

train_dataset = pd.read_csv('E:\\Development Projects\\AI Data\\FashionMSNIT\\data\\train.csv')
train_dataset = train_dataset.head(20060)
train_img_dir = 'E:\\Development Projects\\AI Data\\FashionMSNIT\\data\\train-reduced\\'

test_dataset = pd.read_csv('E:\\Development Projects\\AI Data\\FashionMSNIT\\data\\test.csv')
test_img_dir = 'E:\\Development Projects\\AI Data\\FashionMSNIT\\data\\test\\'


# Load and preprocess training data
def process_train_data():
    train_image = []
    for i in tqdm(range(train_dataset.shape[0])):
        img = image.load_img(train_img_dir + train_dataset['id'][i].astype('str') + '.png', target_size=(28, 28, 1),
                             color_mode='grayscale')
        img = image.img_to_array(img)
        img = img / 255  # convert the image values from 0-255 to 0-1
        train_image.append(img)
    train_X = np.array(train_image)
    train_y = train_dataset['label'].values

    # perform one hot encode
    train_y = to_categorical(train_y, num_classes=10)

    return train_X, train_y


# Load and preprocess testing data
def process_test_data():
    test_image = []
    for i in tqdm(range(test_dataset.shape[0])):
        img = image.load_img(test_img_dir + test_dataset['id'][i].astype('str') + '.png', target_size=(28, 28, 1),
                             color_mode='grayscale')
        img = image.img_to_array(img)
        img = img / 255  # convert the image values from 0-255 to 0-1
        test_image.append(img)
    test_X = np.array(test_image)

    return test_X


def save_prediction(prediction):
    submit_prediction = pd.read_csv('utilities/submit-prediction-template.csv')
    submit_prediction['label'] = prediction
    submit_prediction.to_csv('utilities/submit-prediction.csv', header=True, index=False)
