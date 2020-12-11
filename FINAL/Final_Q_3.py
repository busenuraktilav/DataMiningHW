import pandas as pd
from random import randrange, seed
from sklearn.naive_bayes import GaussianNB
from itertools import chain
import statistics

#data is preprocessed according to zero mean unit variance
def preprocess(dataset):

    means = (list(dataset.mean()))
    stds = (list(dataset.std()))
    cols = list(dataset.columns)

    for i in range(len(cols)):
        dataset[cols[i]] = dataset[cols[i]].apply(lambda x: (x-means[i])/stds[i])

    return dataset

def cv_split(dataset, folds):

    dataset_split = []
    dataset_cp = dataset
    fold_size = int(dataset_cp.shape[0]/folds)

    for i in range(folds):
        fold = []
        while len(fold) < fold_size:
            k = randrange(dataset_cp.shape[0])
            index = dataset_cp.index[k]
            fold.append(k)
            dataset_cp = dataset_cp.drop(index)

        dataset_split.append(fold)

    return dataset_split


def train_test_fold(dataset, indexes):

    datasets = []

    for i in indexes:
        temp = dataset.iloc[i]
        datasets.append(temp)

    return datasets

def confusion(true, pred):

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(true)):

        if(true[i] == pred[i] and true[i] == 1):
            true_pos += 1

        elif(true[i] == pred[i] and true[i] == 0):
            true_neg += 1

        elif(true[i] != pred[i] and true[i] == 1):
            false_neg += 1

        elif(true[i] != pred[i] and true[i] == 0):
            false_pos += 1

        else:
            print("Oops something went wrong")

    return true_pos, true_neg, false_pos, false_neg


def F1_score(true, pred):

    true_pos, true_neg, false_pos, false_neg = confusion(true, pred)

    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)

    f1 = 2*((precision*recall)/(precision+recall))

    return f1

#combine the fold dataset into one dataset by excluding the train set
def combine_for_train(X_fold_list, exc):

    frames = []
    for i in range(len(X_fold_list)):
        if(i != exc):
            frames.append(X_fold_list[i])

    res = pd.concat(frames)
    return res

#flatten 2d array
def flatten(y_fold_list):
    y_fold_list_flat = []
    #flatten 2d list
    for i in range(len(y_fold_list)):
        y_fold_list_flat.append(list(chain.from_iterable(y_fold_list[i].values.tolist())))

    return y_fold_list_flat


def main():

    seed(1)

    diabetes = pd.read_csv("diabetes.csv")
    y = diabetes[['Outcome']]
    X = diabetes.drop('Outcome', 1)

    X = preprocess(X)

    fold = 10
    indexes = cv_split(diabetes, fold)

    X_fold_list = train_test_fold(X, indexes)
    y_fold_list = train_test_fold(y, indexes)

    model = GaussianNB()

    f1_scores = []

    y_fold_list_flatten = flatten(y_fold_list)
    for i in range(fold):
        X_train = combine_for_train(X_fold_list, i)
        y_train = combine_for_train(y_fold_list, i)

        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_fold_list[i])

        f1 = F1_score(y_fold_list_flatten[i], y_pred)
        f1_scores.append(f1)

    #F1 scores for each fold
    print("\nF1 scores for each fold : \n", f1_scores)
    print("\nAverage F1 score : ", statistics.mean(f1_scores))


if __name__ == '__main__':
    main()