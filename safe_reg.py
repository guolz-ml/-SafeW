import numpy as np

from sklearn import datasets

from sklearn.neighbors import KNeighborsRegressor
from cvxopt import solvers, matrix
from sklearn.metrics import mean_squared_error


def read_libsvm(filename):
    """ Read libsvm data from '.txt' file and split to training data and test data
    :param filename: a string indicates the file name
    :return: training instances (x), training labels (y)
    """

    x, y = datasets.load_svmlight_file(filename)
    x = x.toarray()

    return x, y


def split(train_x, train_y, n_labels=10):

    """ Split labeled data and unlabeled data for SSL task
    :param train_x: all training instances
    :param train_y: all training labels
    :param n_labels: numer of labeled instances
    :return: supervised data (x_sup, y_sup) and unsupervised data (x_unsup, y_unsup) where y_unsup can be used to
    test model performance.
    """

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


def baseline_predict(sup_x, sup_y, unsup_x):
    """This is a  1NN regressor with euclidean distance measure.

        sup_x : array-like
            laeled data matrix with [n_samples, n_features]

        y : array-like
            label  matrix with [n_sample, ]

        unsup_x: array-like
            unlabeled data matrix with [n_samples, n_features]
    """

    knn = KNeighborsRegressor(n_neighbors=1)
    knn.fit(sup_x, sup_y)
    [dist, idx] = knn.kneighbors(unsup_x)

    baseline_prediction = np.zeros((len(unsup_x), 1))
    for t in range(0, len(unsup_x)):
        baseline_prediction[t] = np.mean(
            np.array([sup_y[i] for i in idx[t]]).flatten())

    return baseline_prediction


def fit_estimator(sup_x, sup_y, unsup_x, n_neighbors=3, metric='minkowski'):
    """Provide ssl methods according to self-training

    """
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric)

    knn.fit(sup_x, sup_y)

    [dist, idx] = knn.kneighbors(unsup_x)

    label_l = sup_y

    label_u = np.zeros((len(unsup_x),))
    for t in range(0, len(unsup_x)):
        label_u[t] = np.mean(np.array([sup_y[i] for i in idx[t]]).flatten())

    y = np.hstack((label_l, label_u))
    [dist, idx] = knn.kneighbors(unsup_x)

    for i in range(0, 5):
        label_u = np.zeros((len(unsup_x),))
        for t in range(0, len(unsup_x)):
            label_u[t] = np.mean(np.array([y[i] for i in idx[t]]).flatten())
        y = np.hstack((label_l, label_u))

    return label_u


def fit_pred(candidate_prediction=None, baseline_prediction=None):
    """

    Parameters
    ----------
    candidate_prediction :array-like, optical(default=None)
        a matrix with size instance_num * candidate_num . Each
        column vector of candidate_prediction is a candidate regression
        result.

    baseline_prediction :array-like, optical(default=None)
        a column vector with length instance_num. It is the regression
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

    H = np.dot(candidate_prediction.T, candidate_prediction) * 2
    f = -2 * np.dot(candidate_prediction.T, baseline_prediction)
    Aeq = np.ones((1, candidate_num))
    beq = 1.0

    lb = np.zeros((candidate_num, 1))
    ub = np.ones((candidate_num, 1))
    h = np.vstack((lb, ub))
    G_lb = -1 * np.eye(candidate_num, candidate_num)
    G_ub = np.eye(candidate_num, candidate_num)
    G = np.vstack((G_lb, G_ub))

    sln = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h),
                     matrix(Aeq), matrix(beq))

    x_value = sln['x'] # weights for base learners

    safe_prediction = np.zeros((baseline_prediction.shape[0], 1))
    for i in range(0, candidate_num):
        safe_prediction[:, 0] = safe_prediction[:,
                                0] + x_value[i] * candidate_prediction[:, i]

    return safe_prediction, x_value

def cal_mse(y_pred, y_truth):
    return mean_squared_error(y_truth, y_pred)


if __name__ == "__main__":

    #construct dataset
    train_x, train_y = read_libsvm("data/abalone.txt")
    sup_x, sup_y, unsup_x, unsup_y = split(train_x, train_y)

    #base predictions
    candidate_prediction1 = fit_estimator(
        sup_x, sup_y, unsup_x, n_neighbors=3, metric='euclidean')
    candidate_prediction2 = fit_estimator(
        sup_x, sup_y, unsup_x, n_neighbors=3, metric='cosine')
    candidate_prediction3 = fit_estimator(
        sup_x, sup_y, unsup_x, n_neighbors=3)

    candidate_prediction = np.stack(
        [candidate_prediction1, candidate_prediction2, candidate_prediction3], axis=1)
    baseline_prediction = baseline_predict(sup_x, sup_y, unsup_x)

    #safe prediction
    safe_prediction, weights = fit_pred(candidate_prediction, baseline_prediction)

    print("Mse of baseline prediction: ", cal_mse(baseline_prediction, unsup_y))
    print("Mse of candidate prediction1: ", cal_mse(candidate_prediction1, unsup_y))
    print("Mse of candidate prediction2: ", cal_mse(candidate_prediction2, unsup_y))
    print("Mse of candidate prediction3: ", cal_mse(candidate_prediction3, unsup_y))
    print("Mse of safe prediction: ", cal_mse(safe_prediction, unsup_y))
    print("Weights of base learners: ", weights)
