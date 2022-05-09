from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


# CNN model for multiclass predictions
def convNet_model(num_classes):
    visible_layer = Input(shape=(28, 28, 1))
    conv_layer_1 = Conv2D(185, kernel_size=(12, 12), activation='relu', padding='same')(visible_layer)
    pooling_layer_1 = MaxPooling2D(pool_size=(9, 9))(conv_layer_1)
    conv_layer_2 = Conv2D(115, kernel_size=(7, 7), activation='relu', padding='same')(pooling_layer_1)
    conv_layer_3 = Conv2D(75, kernel_size=(5, 5), activation='relu', padding='same')(conv_layer_2)
    pooling_layer_2 = MaxPooling2D(pool_size=(3, 3))(conv_layer_3)
    transform_matrix_to_vector = Flatten()(pooling_layer_2)
    fc_layer_1 = Dense(154, activation='relu')(transform_matrix_to_vector)
    dropout_layer = Dropout(rate=0.2)(fc_layer_1)
    fc_layer_2 = Dense(80, activation='relu')(dropout_layer)
    fc_layer_3 = Dense(54, activation='relu')(fc_layer_2)
    fc_layer_4 = Dense(25, activation='relu')(fc_layer_3)
    output_layer = Dense(num_classes, activation='softmax')(fc_layer_4)

    model = Model(visible_layer, output_layer)

    plot_model(model, to_file='utilities/convNet.png', show_shapes=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model