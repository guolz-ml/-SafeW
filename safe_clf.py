import numpy as np
import scipy.io as sio

from sklearn.neighbors import KNeighborsClassifier

import cvxpy as cp

#get a demo dataset, you can use your own dataset
def read_data(filename="rea.mat"):
    mat = sio.loadmat(filename)
    train_x = mat["train_X"]  # 8096 instances * 256  features
    train_y = mat["train_y"].reshape(-1)  # 0/1, binary classification
    test_x = mat["test_X"]  # 1920 * 256
    test_y = mat["test_y"].reshape(-1)

    train_y[np.where(train_y == 0)] = -1
    test_y[np.where(test_y == 0)] = -1

    return train_x, train_y, test_x, test_y

def split(train_x, train_y, n_labels=100):
    np.random.seed(5)

    n = train_x.shape[0]
    rand_index = np.random.permutation(np.arange(n))

    labels_index = rand_index[:n_labels]
    unlabels_index = rand_index[n_labels:]

    x_sup = train_x[labels_index, :]
    y_sup = train_y[labels_index]

    x_unsup = train_x[unlabels_index, :]
    y_unsup = train_y[unlabels_index]

    return x_sup, y_sup, x_unsup, y_unsup


#obatin baseline supervised prediction with supervised data only
def baseline_predict(sup_x, sup_y, test_x):
    """This is a  1NN classifier with euclidean distance measure.

        Return: prediction on test data
    """

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(sup_x, sup_y)
    [dist, idx] = knn.kneighbors(test_x)

    baseline_prediction = np.zeros((len(test_x), 1))
    for t in range(0, len(test_x)):
        baseline_prediction[t] = np.mean(
            np.array([sup_y[i] for i in idx[t]]).flatten())

    baseline_prediction[np.where(baseline_prediction >= 0)] = 1
    baseline_prediction[np.where(baseline_prediction < 0)] = -1

    return baseline_prediction

#obtain ssl precitions, it is just a demo, you can choose base learner by your self.
def fit_estimator(sup_x, sup_y, unsup_x, test_x, n_neighbors=3, metric='minkowski'):
    """Provide a ssl methods according to self-training

    """

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

    knn.fit(sup_x, sup_y)

    [dist, idx] = knn.kneighbors(unsup_x)

    label_l = sup_y

    label_u = np.zeros((len(unsup_x),))
    for t in range(0, len(unsup_x)):
        label_u[t] = np.mean(np.array([sup_y[i] for i in idx[t]]).flatten())

    x = np.vstack((sup_x, unsup_x))
    y = np.hstack((label_l, label_u))

    [dist, idx] = knn.kneighbors(unsup_x)

    # assign pesudo-label for unlabel instances
    for i in range(0, 5):
        label_u = np.zeros((len(unsup_x),))
        for t in range(0, len(unsup_x)):
            label_u[t] = np.mean(np.array([y[i] for i in idx[t]]).flatten())
        y = np.hstack((label_l, label_u))

    [dist, idx] = knn.kneighbors(test_x)

    prediction = np.zeros((len(test_x),))
    for t in range(0, len(test_x)):
        prediction[t] = np.mean(
            np.array([y[i] for i in idx[t]]).flatten())

    prediction[np.where(prediction >= 0)] = 1
    prediction[np.where(prediction < 0)] = -1

    return prediction


#SafeW
def fit_pred(candidate_prediction=None, baseline_prediction=None):
    """

    Parameters
    ----------
    candidate_prediction :array-like, optical(default=None)
        a matrix with size instance_num * candidate_num . Each
        column vector of candidate_prediction is a candidate classification
        result.

    baseline_prediction :array-like, optical(default=None)
        a column vector with length instance_num. It is the classification
        result of the baseline method.

    Return
    ------
    Safe_prediction : array-like
        a predictive result
    x_values: array-like
        weight of each base learner

    """

    if candidate_prediction is None:
        raise ValueError("Please provide candidate prediction or "
                         "call the function that generates the prediction "
                         "result in this algorithm.")
    else:
        if baseline_prediction is None and baseline_prediction is None:
            raise ValueError("Please provide candidate prediction or "
                             "call the function that generates the prediction result "
                             "in this algorithm.")

    candidate_num = candidate_prediction.shape[1]

    H = np.dot(baseline_prediction.T, candidate_prediction)
    x = cp.Variable(candidate_num)
    lb = np.ones((1, candidate_num))
    objective = cp.Minimize(H@x)
    constraints = [x>=0, lb@x==1]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    x_value = x.value

    safe_prediction = np.zeros((baseline_prediction.shape[0], 1))
    for i in range(0, candidate_num):
        safe_prediction[:, 0] = safe_prediction[:,0] + x_value[i] * candidate_prediction[:, i]

    safe_prediction[np.where(safe_prediction >= 0)] = 1
    safe_prediction[np.where(safe_prediction < 0)] = -1

    return safe_prediction, x_value

def cal_acc(y_pred, y_truth):
    tot = 0
    for i in range(y_pred.shape[0]):
        if(y_pred[i] == y_truth[i]):
            tot += 1
    return 1.0 * tot / y_pred.shape[0]

if __name__ == "__main__":

    #dataset construct
    train_x, train_y, test_x, test_y = read_data()
    sup_x, sup_y, unsup_x, unsup_y = split(train_x, train_y)


    # base predictions
    candidate_prediction1 = fit_estimator(
        sup_x, sup_y, unsup_x, test_x, n_neighbors=3, metric='euclidean')

    candidate_prediction2 = fit_estimator(
        sup_x, sup_y, unsup_x, test_x, n_neighbors=3, metric='cosine')

    candidate_prediction3 = fit_estimator(
        sup_x, sup_y, unsup_x, test_x, n_neighbors=3)

    candidate_prediction = np.stack(
        [candidate_prediction1, candidate_prediction2, candidate_prediction3],axis=1)

    baseline_prediction = baseline_predict(sup_x, sup_y, test_x)

    #compute safe prediction
    safe_prediction, weights = fit_pred(candidate_prediction, baseline_prediction)


    print("Accuracy of candidate prediction 1: ", cal_acc(candidate_prediction1, test_y))

    print("Accuracy of candidate prediction 2: ", cal_acc(candidate_prediction2, test_y))

    print("Accuracy of candidate prediction 3: ", cal_acc(candidate_prediction3, test_y))

    print("Accuracy of baseline prediction: ", cal_acc(baseline_prediction, test_y))

    print("Accuracy of safe prediction: ", cal_acc(safe_prediction, test_y))

    print("Weights for base learners: ", weights)
