import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_ind
import scipy
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as ts
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import CategoricalNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def ks_test(train, test):
    rows, cols = train.shape
    d = np.zeros((cols, 1))
    pval = np.zeros((cols, 1))
    '''ks test'''
    for i in range(cols):
        d[i], pval[i] = stats.ks_2samp(train.iloc[:, i], test.iloc[:, i])
    if pval.all() < 0.05:
        print("Reject the Null Hypothesis.")
    else:
        print("Accept the Null Hypothesis.")
    return pval


def kl_test(train, test):
    rows, cols = train.shape
    KL = np.zeros((cols, 1))
    total_time = 10 * int(len(train) / len(test))
    eta = 1e-6
    for time in range(total_time):
        subsample = np.random.randint(0, len(train), (len(test),))
        for i in range(cols):
            KL[i] += scipy.stats.entropy(train.iloc[subsample, i], test.iloc[:, i] + eta)
    KL /= total_time
    if KL.all() < 0.05:
        print("Reject the Null Hypothesis.")
    else:
        print("Accept the Null Hypothesis.")
    return KL


def ftest(train, test):
    '''F检验样本总体方差是否相等'''
    F = np.var(train) / np.var(test)
    v1 = len(train) - 1
    v2 = len(test) - 1
    p_val = 1 - 2 * abs(0.5 - stats.f.cdf(F, v1, v2))
    if p_val.all() < 0.05:
        print("Reject the Null Hypothesis.")
        equal_var = False
    else:
        print("Accept the Null Hypothesis.")
        equal_var = True
    return equal_var


def ttest_ind_fun(train, test):
    '''t检验独立样本所代表的两个总体均值是否存在差异'''
    equal_var = ftest(train, test)
    ttest, pval = ttest_ind(train, test, equal_var=equal_var)
    if pval.all() < 0.05:
        print("Reject the Null Hypothesis.")
    else:
        print("Accept the Null Hypothesis.")
    return pval


def svm_test(train, test, test_size=0.3, kernel_method='rbf', confidence_score=True, cv=5):
    '''
    Based on SVM
    :param train: train data
    :param test: test data
    :param test_size: the size of data that used to measure how splittable of dataset
    :param kernel_method: kernel method of svm (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’)
    :param confidence_score: the confidence to the data
    :return: confidence_score
    '''
    # generate label for data from train or test dataset
    y_train = np.zeros(len(train), dtype=np.int)
    y_test = np.ones(len(test), dtype=np.int)
    X = np.concatenate([train, test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    # split data
    X_train, X_test, y_train, y_test = ts(X, y, test_size=test_size)

    # create the model
    clf = svm.SVC(kernel=kernel_method)
    scores = cross_val_score(clf, X, y, cv=cv)
    score = scores.mean()
    clf.fit(X_train, y_train)
    # calculate the similarity of train and test parts
    similarity = 0.5 - np.abs(score - 0.5)
    # scale the similarity into 0~1
    similarity *= 200  # similarity = similarity * 0.1 *10 * 100
    # similarity = similarity
    print('The similarity of train and test data is: %.2f%% ' % similarity)
    # return confidence score
    if confidence_score:
        return clf.decision_function(train), clf.decision_function(test)


def knn_test(train, test, test_size=0.3, algorithm='auto', cv=5):
    '''
    Based on KNeighbors
    :param train: train data
    :param test: test data
    :param test_size: the size of data that used to measure how splittable of dataset
    :param algorithm: algorithm of KNeighbors (refer to the sklearn document)
    :return:
    '''
    # generate label for data from train or test dataset
    y_train = np.zeros(len(train), dtype=np.int)
    y_test = np.ones(len(test), dtype=np.int)
    X = np.concatenate([train, test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    # split data
    X_train, X_test, y_train, y_test = ts(X, y, test_size=test_size)

    # create the model
    neigh = KNeighborsClassifier(n_neighbors=2, algorithm=algorithm)
    scores = cross_val_score(neigh, X, y, cv=cv)
    score = scores.mean()
    neigh.fit(X_train, y_train)
    # calculate the similarity of train and test parts
    similarity = 0.5 - np.abs(score - 0.5)
    # scale the similarity into 0~1
    similarity *= 200  # similarity = similarity * 0.1 *10 * 100
    # similarity = similarity
    print('The similarity of train and test data is: %.2f%% ' % similarity)


def kmeans_test(train, test, test_size=0.3, algorithm='auto'):
    '''
    Based on KMeans
    :param train: train data
    :param test: test data
    :param test_size: the size of data that used to measure how splittable of dataset
    :param algorithm: algorithm of KMeans (refer to the sklearn document)
    :return:
    '''
    # generate label for data from train or test dataset
    y_train = np.zeros(len(train), dtype=np.int)
    y_test = np.ones(len(test), dtype=np.int)
    X = np.concatenate([train, test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    # split data
    X_train, X_test, y_train, y_test = ts(X, y, test_size=test_size)

    # create the model
    kmeans = KMeans(n_clusters=2, algorithm=algorithm)
    kmeans.fit(X_train)
    pred = kmeans.predict(X_test)
    score = (pred == y_test).mean()
    # calculate the similarity of train and test parts
    similarity = 0.5 - np.abs(score - 0.5)
    # scale the similarity into 0~1
    similarity *= 200  # similarity = similarity * 0.1 *10 * 100
    # similarity = similarity
    print('The similarity of train and test data is: %.2f%% ' % similarity)


def ridge_test(train, test, test_size=0.3, solver='auto', normalize=True, confidence_score=False, cv=5):
    '''
    Based on RidgeClassifier
    :param train: train data
    :param test: test data
    :param test_size: the size of data that used to measure how splittable of dataset
    :param solver: solver of KMeans (refer to the sklearn document)
    :param normalize: normalize data or not
    :param confidence_score: the confidence to the data
    :return: confidence_score
    '''
    # generate label for data from train or test dataset
    y_train = np.zeros(len(train), dtype=np.int)
    y_test = np.ones(len(test), dtype=np.int)
    X = np.concatenate([train, test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    # split data
    X_train, X_test, y_train, y_test = ts(X, y, test_size=test_size)

    # create the model
    clf = RidgeClassifier(solver=solver, normalize=normalize)
    scores = cross_val_score(clf, X, y, cv=cv)
    score = scores.mean()
    clf.fit(X_train, y_train)

    # calculate the similarity of train and test parts
    similarity = 0.5 - np.abs(score - 0.5)
    # scale the similarity into 0~1
    similarity *= 200  # similarity = similarity * 0.1 *10 * 100
    # similarity = similarity
    print('The similarity of train and test data is: %.2f%% ' % similarity)
    # return confidence score
    if confidence_score:
        return clf.decision_function(train), clf.decision_function(test)


def LR_test(train, test, test_size=0.3, solver='lbfgs', confidence_score=False, max_iter=1000, cv=5):
    '''
    Based on LogisticRegression
    :param max_iter: the max iteration
    :param train: train data
    :param test: test data
    :param test_size: the size of data that used to measure how splittable of dataset
    :param solver: solver of KMeans (refer to the sklearn document)
    :param confidence_score: the confidence to the data
    :return: confidence_score
    '''
    # generate label for data from train or test dataset
    y_train = np.zeros(len(train), dtype=np.int)
    y_test = np.ones(len(test), dtype=np.int)
    X = np.concatenate([train, test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    # split data
    X_train, X_test, y_train, y_test = ts(X, y, test_size=test_size)

    # create the model
    clf = LogisticRegression(solver=solver, max_iter=max_iter)
    scores = cross_val_score(clf, X, y, cv=cv)
    score = scores.mean()
    clf.fit(X_train, y_train)

    # calculate the similarity of train and test parts
    similarity = 0.5 - np.abs(score - 0.5)
    # scale the similarity into 0~1
    similarity *= 200  # similarity = similarity * 0.1 *10 * 100
    # similarity = similarity
    print('The similarity of train and test data is: %.2f%% ' % similarity)
    # return confidence score
    if confidence_score:
        return clf.decision_function(train), clf.decision_function(test)


def CNB_test(train, test, test_size=0.3, normalize=True, confidence_score=False, cv=5):
    '''
    Based on ComplementNB
    :param normalize: normalize data or not
    :param train: train data
    :param test: test data
    :param test_size: the size of data that used to measure how splittable of dataset
    :param confidence_score: the confidence to the data
    :return: confidence_score, the probability of the samples for each class in the model.
    '''
    # generate label for data from train or test dataset
    y_train = np.zeros(len(train), dtype=np.int)
    y_test = np.ones(len(test), dtype=np.int)
    X = np.concatenate([train, test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    # split data
    X_train, X_test, y_train, y_test = ts(X, y, test_size=test_size)

    # create the model
    clf = ComplementNB(norm=normalize)
    scores = cross_val_score(clf, X, y, cv=cv)
    score = scores.mean()
    clf.fit(X_train, y_train)

    # calculate the similarity of train and test parts
    similarity = 0.5 - np.abs(score - 0.5)
    # scale the similarity into 0~1
    similarity *= 200  # similarity = similarity * 0.1 *10 * 100
    # similarity = similarity
    print('The similarity of train and test data is: %.2f%% ' % similarity)
    # return confidence score
    if confidence_score:
        return clf.predict_proba(train), clf.predict_proba(test)


def DecTree_test(train, test, test_size=0.3, criterion='gini', max_depth=None, confidence_score=False, cv=5):
    '''
    Based on DecisionTreeClassifier
    :param max_depth: The maximum depth of the tree
    :param criterion: {“gini”, “entropy”}
    :param train: train data
    :param test: test data
    :param test_size: the size of data that used to measure how splittable of dataset
    :param confidence_score: the confidence to the data
    :return: confidence_score, the probability of the samples for each class in the model.
    '''
    # generate label for data from train or test dataset
    y_train = np.zeros(len(train), dtype=np.int)
    y_test = np.ones(len(test), dtype=np.int)
    X = np.concatenate([train, test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    # split data
    X_train, X_test, y_train, y_test = ts(X, y, test_size=test_size)

    # create the model
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    scores = cross_val_score(clf, X, y, cv=cv)
    score = scores.mean()

    clf.fit(X_train, y_train)


    # calculate the similarity of train and test parts
    similarity = 0.5 - np.abs(score - 0.5)
    # scale the similarity into 0~1
    similarity *= 200  # similarity = similarity * 0.1 *10 * 100
    # similarity = similarity
    print('The similarity of train and test data is: %.2f%% ' % similarity)
    # return confidence score
    if confidence_score:
        return clf.predict_proba(train), clf.predict_proba(test)


def RT_test(train, test, test_size=0.3, criterion='gini', max_depth=None, n_estimators=100, confidence_score=False, cv=5):
    '''
    Based on RandomForestClassifier
    :param n_estimators: The number of trees in the forest.
    :param max_depth: The maximum depth of the tree
    :param criterion: {“gini”, “entropy”}
    :param train: train data
    :param test: test data
    :param test_size: the size of data that used to measure how splittable of dataset
    :param confidence_score: the confidence to the data
    :return: confidence_score, the probability of the samples for each class in the model.
    '''
    # generate label for data from train or test dataset
    y_train = np.zeros(len(train), dtype=np.int)
    y_test = np.ones(len(test), dtype=np.int)
    X = np.concatenate([train, test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    # split data
    X_train, X_test, y_train, y_test = ts(X, y, test_size=test_size)

    # create the model
    clf = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators)
    scores = cross_val_score(clf, X, y, cv=cv)
    score = scores.mean()
    clf.fit(X_train, y_train)
    # score = clf.score(X_test, y_test,)

    # calculate the similarity of train and test parts
    similarity = 0.5 - np.abs(score - 0.5)
    # scale the similarity into 0~1
    similarity *= 200  # similarity = similarity * 0.1 *10 * 100
    # similarity = similarity
    print('The similarity of train and test data is: %.2f%% ' % similarity)
    # return confidence score
    if confidence_score:
        return clf.predict_proba(train), clf.predict_proba(test)


if __name__ == '__main__':
    from sklearn import datasets

    boston = datasets.load_boston()
    # concat X and Y and get new X matrix
    X = boston.data
    y = boston.target.reshape(-1, 1)
    # generate our new labels(from train or test dataset)
    print(len(X))
    X_train, X_test = ts(X, test_size=0.3)

    svm_test(X_train, X_test)
    knn_test(X_train, X_test)
    kmeans_test(X_train, X_test)
    LR_test(X_train, X_test, max_iter=1e4)
    ridge_test(X_train, X_test)
    CNB_test(X_train, X_test)
    DecTree_test(X_train, X_test)
    RT_test(X_train, X_test, n_estimators=3, max_depth=5, confidence_score=True)
