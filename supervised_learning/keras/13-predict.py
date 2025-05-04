#!/usr/bin/env python3
import tensorflow.keras as K

def one_hot(Y, classes):
    return K.utils.to_categorical(Y, classes)

def load_model(path):
    return K.models.load_model(path)

def predict(network, data):
    return network.predict(data)

def reshape(flat_images):
    return flat_images.reshape((flat_images.shape[0], -1))

def argmax(predictions):
    return K.backend.eval(K.backend.argmax(predictions, axis=1))

if __name__ == '__main__':
    # Use Keras MNIST loader (built-in dataset instead of npz file)
    (_, _), (X_test, Y_test) = K.datasets.mnist.load_data()
    X_test = X_test.astype('float32') / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1))

    network = load_model('network2.keras')
    Y_pred = predict(network, X_test)

    print(Y_pred)
    print(argmax(Y_pred))  # ✅ Matches desired output
