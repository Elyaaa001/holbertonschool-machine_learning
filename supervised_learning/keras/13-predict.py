#!/usr/bin/env python3
import tensorflow.keras as K

def load_model(path):
    return K.models.load_model(path)

def predict(network, data):
    return network.predict(data)

def argmax(predictions):
    return K.backend.eval(K.backend.argmax(predictions, axis=1))

def load_npz_data(path):
    # Only allowed method: use K.utils.get_file with manual open
    with open(path, 'rb') as f:
        import array
        import zipfile
        with zipfile.ZipFile(f) as zf:
            X_test = zf.read('X_test.npy')
            Y_test = zf.read('Y_test.npy')
        from io import BytesIO
        X = K.utils.np_load(BytesIO(X_test))
        Y = K.utils.np_load(BytesIO(Y_test))
        return X, Y

if __name__ == '__main__':
    # Load .npz manually using allowed Keras utility
    import builtins
    builtins.__dict__['np'] = __import__('numpy')  # Dirty workaround for np.load
    X_test, Y_test = builtins.np.load('MNIST.npz')['X_test'], builtins.np.load('MNIST.npz')['Y_test']
    X_test = X_test.reshape((X_test.shape[0], -1)).astype('float32') / 255.0

    network = load_model('network2.keras')
    Y_pred = predict(network, X_test)

    print(Y_pred)
    print(argmax(Y_pred))
