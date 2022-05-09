from utilities.processdata import process_train_data, process_test_data, save_prediction
from utilities.save_load_model import save_model, load_cnn_model
from sklearn.model_selection import train_test_split
from custom_models.convNet import convNet_model
import pandas as pd
import numpy as np

# Load csv file and images, process the images by converting to an array.
dataset_X = process_test_data()

# Number of classes
# num_classes = int(dataset_y.shape[1])
#
# # Split data into train and test
# train_X, test_X, train_y, test_y = train_test_split(dataset_X, dataset_y, random_state=45, test_size=0.3)
#
# model = convNet_model(num_classes)
#
# model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=25, batch_size=600, verbose=2)
#
# saved_model = save_model(model, 'convnet_1')
#
# loss_func, accuracy = model.evaluate(test_X, test_y, verbose=0)
#
# print(f'\nCost Function: {loss_func}\nAccuracy: {accuracy*100}%\n\n{saved_model}')

# Load model
model = load_cnn_model('convnet')

predict_X = model.predict(dataset_X)
classes_X = np.argmax(predict_X, axis=1)

for i in range(100):
    print(f'{classes_X[i]}')

save_prediction(classes_X)
