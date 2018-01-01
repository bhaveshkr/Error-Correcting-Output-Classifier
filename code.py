"""ERROR CORRECTING OUTPUT CLASSIFIER (ECOC)
"""
__author__ = 'Bhavesh Kumar'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Hamming distance function
def hamming_distance(vector1, vector2):
    return np.sum(np.abs(vector1-vector2))/n


# sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))


# equation 1 for sigmoid
def delta_w(w_vector, m_vector, q, x):
    return 0.005*(np.abs(w_vector-m_vector))*(0.5-q)*(q-q**2)*x


# Error Correcting Code Classifier - Initialization part
n = 7 # no. of bits
err = 0.48
max_itr = 150
threshold = 0.5

DataFrame = pd.read_csv('ecoli.csv', header=None)
DataMatrix = DataFrame.as_matrix()
New_DataMatrix = DataMatrix[:,0:7]

Q, P = New_DataMatrix.shape # Q- samples, P- Features
class_labels = DataMatrix[:, 7:]
unique_Class_labels = np.unique(class_labels)
C = len(np.unique(class_labels)) # no. of classes
M = np.random.randint(2, size=(C, n))
w = np.random.randint(low=-1, high=1, size=(P, n)) # weight matrix

normalized_M = normalize(New_DataMatrix, norm='l2', axis=0) # normalize 0-1 for sigmoid
X_train, X_test, y_train, y_test = train_test_split(normalized_M, class_labels, test_size = 0.2)
no_of_training_samples = len(X_train)
no_of_testing_samples = len(X_test)

print(w)


# Error Correcting Code Classifier - Training part
iter = 0
prediction_vector = np.zeros((no_of_training_samples, n), dtype=float)
training_error = []
error = 0.0
while error > err or iter < max_itr:
    iter += 1
    err_part1 = [None] * no_of_training_samples
    for j in range(no_of_training_samples):
        Z = np.dot(X_train[j], w.T)
        C_star = y_train[j]
        q = sigmoid(Z)
        for index, eachQ in enumerate(q):
            if eachQ < threshold:
                prediction_vector[j][index] = 0
            elif eachQ >= threshold:
                prediction_vector[j][index] = 1
        index_of_C_start, = np.where(unique_Class_labels == C_star)
        C_star_row = M[index_of_C_start]
        w = w + delta_w(prediction_vector[j], C_star_row, q, X_train[j])
        err_part1[j] = hamming_distance(prediction_vector[j], C_star_row)
    error = np.sum(err_part1)/no_of_training_samples
    training_error.append(error)
    print(error)
print(w)


# Error Correcting Code Classifier - Testing part
def test(data_x, data_y, no_of_samples):
    prediction_vector_1 = [None]*n
    y_predicted = []
    for z in range(no_of_samples):
        Z = np.dot(data_x[z], w.T)
        C_star = data_y[z]
        q = sigmoid(Z)
        for index, eachQ in enumerate(q):
            if eachQ < threshold:
                prediction_vector_1[index] = 0
            elif eachQ >= threshold:
                prediction_vector_1[index] = 1
        distance = []
        for i in range(len(M)):
            distance.append(hamming_distance(prediction_vector_1, M[i]))    
        y_predicted.append(distance.index(min(distance)))
    return y_predicted


# calculate confusion matrix
y_train_predicted = test(X_train, y_train, no_of_training_samples)
y_test_predicted = test(X_test, y_test, no_of_testing_samples)

class_map = {"cp":0, "im":1, "imS":2, "imL":3, "imU":4, "om":5, "omL":6, "pp":7}
y_train_new = []
y_test_new = []
for value in y_train:
    y_train_new.append(class_map[value[0]])
for value in y_test:
    y_test_new.append(class_map[value[0]])

train_matrix = confusion_matrix(y_train_new, y_train_predicted)
test_matrix = confusion_matrix(y_test_new, y_test_predicted)


# plot confusion matrix
def plot_cm(data, range_val, title):
    ax = plt.axes()
    cm_train = pd.DataFrame(data, index = range(range_val), columns = range(range_val))
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(cm_train, annot=True,annot_kws={"size": 16}, ax=ax)# font size
    ax.set_title(title)
    plt.show()
    
plot_cm(train_matrix, 8, 'Training - Confusion Matrix')
plot_cm(test_matrix, 7, 'Testing - Confusion Matrix')


# plot chart - Iteration versus Error
x_axis = [i for i,_ in enumerate(training_error)]
# print(x_axis)
plt.plot(x_axis, training_error)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()



