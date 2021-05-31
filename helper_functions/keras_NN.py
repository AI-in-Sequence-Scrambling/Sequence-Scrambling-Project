from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.utils import np_utils
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


# Method to initialize a neural network model according to the provided parameters
def create_model (layer_depth, layer_architecture, input_activation_function, input_shape, input_neurons, dropout, hidden_neurons, hidden_activation_function, 
output_activation_function, output_neurons):
    
    #Create a new empty aNN
    model = models.Sequential()
    
    # Input - Layer
    model.add(layers.Dense(input_neurons, activation = input_activation_function, input_shape=(input_shape, )))
    
    # Hidden - Layers
    for i in range (0,layer_depth):
        for j in layer_architecture:
            if j == 'Dense':
                model.add(layers.Dense(hidden_neurons, activation = hidden_activation_function))
            if j == 'Dropout':
                model.add(layers.Dropout(dropout, noise_shape=None, seed=None))
            if j == 'Conv2D':
                model.add(layers.Conv2D())
            if j == 'MaxPooling2D':
                model.add(layers.MaxPooling2D())

            print('Layer ' + str(j) + ' added to the network')

    # Output- Layer
    model.add(layers.Dense(output_neurons, activation = output_activation_function))
    
    return model
    

# Method to train and predict on the neural network classifier
def train_model (model, optimizer, loss, metric, epochs, batch_size, X_train, y_train, X_test, y_test):

    # Compiling the model
    model.compile(
     optimizer = optimizer,
     loss = loss,
     metrics = [metric]
    )
    
    # Fit the model
    results = model.fit(
     X_train, y_train,
     batch_size = batch_size,
     epochs = epochs,
     validation_data = (X_test, y_test)
    )
    print("Test-Accuracy:", np.mean(results.history["val_accuracy"]))
    
    # Perform predictions on test data
    print('Perform prediction on test-data')
    predictions = model.predict(X_test)
    
    return model, predictions, results