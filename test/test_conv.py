import sys 
sys.path.append('..')
from projet_etu import *
import numpy as np
import pickle as pkl
from utils import *
from loss import *
from activation import *
from projet_etu import *
from convolution import *
# Load data
data = pkl.load(open("../data/usps.pkl", "rb"))

X_train = np.array(data["X_train"], dtype=float).reshape(-1, 16 * 16, 1)
X_test = np.array(data["X_test"], dtype=float).reshape(-1, 16 * 16, 1)
# X_train = np.array(data["X_train"], dtype=float)
# X_test = np.array(data["X_test"], dtype=float)
# X_train = X_train[:, :, np.newaxis]
Y_train = data["Y_train"]
Y_test = data["Y_test"]


def one_hot(y):
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot


Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)

# net = Sequential(
#     [Conv1D(3, 1, 32),
#     MaxPool1D(2, 2),
#     Flatten(),
#     Linear(4064, 100),
#     ReLU(),
#     Linear(100, 10)]
# )

# Create LeNet model
net = Sequential(
    [Conv1D(2,1, 6),
    MaxPool1D(2, 2),
    Conv1D(2, 6, 16),
    MaxPool1D(2, 2),
    Flatten(),
    Linear(1008, 120),
    ReLU(),
    Linear(120, 84),
    ReLU(),
    Linear(84, 10)]
)

loss = CrossEntropyLoss()


Lerror, Lscore= SGD(net, loss, X_train, Y_train, predict= lambda x : np.argmax(net.forward(x),axis = 1), xtest = X_test, ytest = np.argmax(Y_test,axis =1), batch_size=5, max_iter=100, eps =0.001)


# # Compute network output on test set
# output = net.forward(X_test)

# # Compute accuracy
# accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(Y_test, axis=1))
# print(f"Accuracy: {accuracy}")
# # Loss = 0.0921 Accuracy = 0.926

plt.plot(Lerror,label =' loss')
plt.legend()
plt.show()

plt.plot(Lscore,label =' score_test')
plt.legend()
plt.show()