#!/usr/bin/env python3

import tensorflow.keras as K

def one_hot(Y, classes):
    return K.utils.to_categorical(Y, num_classes=classes)

def load_model(path):
    return K.models.load_model(path)

def predict(network, data):
    return network.predict(data)

def reshape(data):
    return data.reshape((data.shape[0], -1))

def argmax(predictions):
    return K.backend.eval(K.backend.argmax(predictions, axis=1))

if __name__ == '__main__':

    (X_train, Y_train), (X_test, Y_test) = K.datasets.mnist.load_data()
    X_test = reshape(X_test.astype('float32') / 255)

    network = load_model('network2.keras')
    Y_pred = predict(network, X_test)

    print(Y_pred)
    print(argmax(Y_pred))
    print(Y_test)
