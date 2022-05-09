from tensorflow.keras.models import Model, load_model


# To learn more, please visit this website: https://machinelearningmastery.com/save-load-keras-deep-learning-models/

def save_model(model: Model, filename):
    model.save(f'utilities/{filename}.h5')

    return f'Model saved as {filename}.h5'


# Load a saved model from your device.
def load_cnn_model(filename):
    model: Model = load_model(f'utilities/{filename}.h5')

    return model
