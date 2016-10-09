from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np

NUM_CLASSES = 10

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    X_train = np.matrix.transpose(X_train)
    X_test = np.matrix.transpose(X_test)
    return (X_train, labels_train), (X_test, labels_test)

def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    XXt=np.dot(X_train,np.matrix.transpose(X_train))
    inverse = np.linalg.inv( XXt + 0.1*np.identity(784))
    return np.dot(inverse,np.dot(X_train,one_hot(y_train)))

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    return np.array([[1 if i == labels_train[k] else 0 for i in range(10)] for k in range(len(labels_train))])
    #actually this way we get the transposition of Y
def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    pred=np.dot(np.matrix.transpose(model),X)
    return [np.argmax(i) for i in np.matrix.transpose(pred)]#get a single array of the prediction y

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    model = train(X_train, labels_train)
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)

    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)


    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
