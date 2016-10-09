from mnist import MNIST
# import sklearn.metrics as metrics
import numpy as np
from numpy.linalg import solve
import scipy
import time

NUM_CLASSES = 10
d=1000
sigma=0.1
G=np.random.normal(scale = sigma, size = (784,d))
b = np.random.random((d,1))*2*3.1415926

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    XXt=np.dot(np.matrix.transpose(X_train),X_train)
    return solve(XXt + reg*np.identity(d),np.dot(np.matrix.transpose(X_train),y_train))


def train_gd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    W = np.zeros((d,10))
    alpha = alpha/X_train.shape[0]
    xtx = np.dot(np.transpose(X_train),X_train)
    h = np.dot(np.transpose(X_train),y_train)
    for i in range(num_iter):
        nabla = 2*np.dot(xtx,W) - 2*h + 2*reg*W
        W = W - alpha*nabla
    return W

def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    W = np.zeros((d,10))
    for i in range(num_iter):
        j = np.random.randint(0,600)
        Xj = X_train[j]
        yj = y_train[j]
        nabla = 2*np.dot(Xj[:, None],Xj,W)-2*np.dot(Xj[:, None],yj) + 2*reg*W
        W = W - nabla*alpha*(1-i/(num_iter + 1)) 

    return W

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    return np.array([[1 if i == labels_train[k] else 0 for i in range(10)] for k in range(len(labels_train))])

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    pred=np.dot(np.matrix.transpose(model),np.transpose(X))
    return [np.argmax(i) for i in np.matrix.transpose(pred)]#get a single array of the prediction y

def phi(X):
    ''' Featurize the inputs using random Fourier features '''
    X=np.cos(np.dot(np.matrix.transpose(G),np.matrix.transpose(X))+b)
    return np.transpose(X)

def accuracy(actual, predicted):
    return (len(actual) - np.count_nonzero(actual - predicted)) / float(len(actual))


if __name__ == "__main__":
    print("d = {0}".format(d))
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)
    X_train, X_test = phi(X_train), phi(X_test)

    # model = train(X_train, y_train, reg=0.1)
    # pred_labels_train = predict(model, X_train)
    # pred_labels_test = predict(model, X_test)
    # print("Closed form solution")
    # print("Train accuracy: {0}".format(accuracy(labels_train, pred_labels_train)))
    # print("Test accuracy: {0}".format(accuracy(labels_test, pred_labels_test)))
    start_time = time.time()
    model = train_gd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=20000)
    print("time taking to train batch gradient descent is :{0}".format(time.time() - start_time))
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Batch gradient descent")
    print("Train accuracy: {0}".format(accuracy(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(accuracy(labels_test, pred_labels_test)))

    # model = train_sgd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=100000)
    # pred_labels_train = predict(model, X_train)
    # pred_labels_test = predict(model, X_test)
    # print("Stochastic gradient descent")
    # print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    # print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
