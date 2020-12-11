import pandas as pd
import math
import matplotlib.pyplot as plt

#min-max normalization algorithm. It takes the min and max value in the training dataset.
def min_max_norm(arr, maxi, mini):

    denom = maxi - mini
    norm_list = []

    for i in arr:
        n = (i-mini)/denom
        norm_list.append(n)

    return(norm_list)


#Turn dataset into column based 2d array to calculate min-max normalization values for each attribute
def column_based_dataset(dataset):

    arr = []

    for i in dataset.columns:
        arr.append(dataset[i])

    return (arr)


#Turn column-based-2d array into raw-based-2d array and vice versa
def transpose(arr):

    trans_arr = []
    for i in range(len(arr[0])):
        temp = []
        for j in range(len(arr)):
            temp.append(arr[j][i])
        trans_arr.append(temp)

    return (trans_arr)


#Get ready for processing the train and test data
def prep_to_process(column_based_2d_arr, column_based_train_data):

    arr = []
    for i in range(len(column_based_2d_arr)):
        arr.append(min_max_norm(column_based_2d_arr[i], max(column_based_train_data[i]), min(column_based_train_data[i])))

    row_based_arr = transpose(arr)

    return (row_based_arr)


#euclidean distance calculation of two points
def euclidean_dist(x,y):

    dist = 0

    for i in range(len(x) - 1):
        dist += (x[i] - y[i])**2

    return (math.sqrt(dist))


#manthatten distance calculation of two points
def manhatten_dist(x, y):

    dist = 0

    for i in range(len(x) - 1):
        dist += abs(x[i] - y[i])

    return (dist)


#find the most close k number of points
def find_most_close(arr, k):
    
    most_close = []
    
    for i in range(k):
        most_close.append(max(arr))

    for i in arr:
        maxi = max(most_close)
        if (i < maxi):
            ind = most_close.index(maxi)
            most_close[ind] = i

    return (most_close)


#find the labels(0 or 1) of that k close points
def labels_most_close(train_arr, k, row_based_train_data):

    indices = []
    most_close = find_most_close(train_arr, k)

    for i in most_close:
        indices.append(train_arr.index(i))

    labels = []

    for i in indices:
        labels.append(row_based_train_data[i][len(row_based_train_data[i])-1])


    return (labels)


#for k close points vote for the most frequent label
def majority_voting(labels):

    class0 = 0
    class1 = 0

    for i in labels:

        if(i == 0):
            class0 += 1

        elif(i == 1):
            class1 += 1

    if(class0 > class1):
        return (0)
    else:
        return (1)


#for k close points vote for according to label's frequency and weight(distance)
def dist_weighted_voting(labels, close):

    weights = []
    for i in close:
        if(i == 0): #if two points are in the same place, then assign it to very small value. (while calculating weight, you cannot divide by zero)
            i = 0.1
        weights.append(1/i)

    class0 = 0
    class1 = 0

    for i in range(len(labels)):
        if(labels[i] == 0):
            class0 += weights[i]
        elif(labels[i] == 1):
            class1 += weights[i]

    if(class0 > class1):
        return (0)
    else:
        return (1)


#Main function to find the nearest neighbours
def knn(train_data, test_data, distance_type, k, voting_type):

    distances = []

    if(distance_type == "e"):

        for i in test_data:
            temp = []
            for j in train_data:
                temp.append(euclidean_dist(i, j))

            distances.append(temp)

    elif(distance_type == "m"):

            for i in test_data:
                temp = []
                for j in train_data:
                    temp.append(manhatten_dist(i, j))

                distances.append(temp)

    else:
        print("Invalid distance type")

    predicted_labels = []

    for i in distances:
        close = find_most_close(i, k)
        train_label = labels_most_close(i, k, train_data)

        if(voting_type == "majority"):
            predicted_labels.append(majority_voting(train_label))

        elif(voting_type == "distance weight"):
            predicted_labels.append(dist_weighted_voting(train_label, close))

        else:
            print("Invalid voting type")

    return (predicted_labels)


#gives the accuracy score
def accuracy(predicted_label, row_based_test_data):

    col_based_test_data = transpose(row_based_test_data)
    count = 0

    for i in range(len(predicted_label)):
        if(predicted_label[i] == col_based_test_data[len(col_based_test_data)-1][i]):
            count += 1

    acc = count/len(predicted_label)

    return (acc)


#draws an accuracy graph
def accuracy_graph(x, y, z, t):

    k = range(1, 35)
    plt.plot(k, x, label="manhatten distance - majority voting")
    plt.plot(k, y, label="manhatten distance - distance weight voting")
    plt.plot(k, z, label="euclidean distance - majority voting")
    plt.plot(k, t, label="euclidean distance - distance weight voting")
    plt.xlabel('different k values')
    plt.ylabel('accuracy')
    plt.title('Accuracy graph')
    plt.legend()
    plt.show()

#It is messy function but it calculates for every k values and draws a accuracy graph
def calculate_results_all_k(test_set, train_set):

    row_based_normed_test_data = prep_to_process(transpose(test_set), train_set)
    row_based_normed_train_data = prep_to_process(train_set, train_set)

    a, b, c, d = [], [], [], []

    # calculate predicted labels for given test set with different parameter settings
    for i in range(1, len(row_based_normed_train_data)):
        a.append(knn(row_based_normed_train_data, row_based_normed_test_data, "m", i, "majority"))
        b.append(knn(row_based_normed_train_data, row_based_normed_test_data, "m", i, "distance weight"))
        c.append(knn(row_based_normed_train_data, row_based_normed_test_data, "e", i, "majority"))
        d.append(knn(row_based_normed_train_data, row_based_normed_test_data, "e", i, "distance weight"))

    x, y, z, t = [], [], [], []

    # calculate accuracy label according to the output of predicted labels
    for i in range(len(a)):
        x.append(accuracy(a[i], test_set))
        y.append(accuracy(b[i], test_set))
        z.append(accuracy(c[i], test_set))
        t.append(accuracy(d[i], test_set))

    accuracy_graph(x, y, z, t)


def find_knn_accuracy(test_set, train_set, k, distance, voting):

    row_based_normed_test_data = prep_to_process(transpose(test_set), train_set)
    row_based_normed_train_data = prep_to_process(train_set, train_set)

    expected_labels = (transpose(test_set))[2]
    print("\nExpected labels: \n\n", expected_labels)


    if(distance == "e" and voting == "distance weight"):

        a = knn(row_based_normed_train_data, row_based_normed_test_data, "e", k, "distance weight")
        x = accuracy(a, test_set)
        print("\nk = ", k, ", calculated based on euclidean distance and distance weight voting\n\n", a)
        print("\nAccuracy level: ", x)

    elif(distance == "m" and voting == "distance weight"):

        a = knn(row_based_normed_train_data, row_based_normed_test_data, "m", k, "distance weight")
        x = accuracy(a, test_set)
        print("\nk = ", k, ", calculated based on manhatten distance and distance weight voting\n\n", a)
        print("\nAccuracy level: ", x)

    elif(distance == "e" and voting == "majority"):

        a = knn(row_based_normed_train_data, row_based_normed_test_data, "e", k, "majority")
        x = accuracy(a, test_set)
        print("\nk = ", k, ", calculated based on euclidean distance and majority voting\n\n", a)
        print("\nAccuracy level: ", x)

    elif(distance == "m" and voting == "majority"):

        a = knn(row_based_normed_train_data, row_based_normed_test_data, "m", k, "majority")
        x = accuracy(a, test_set)
        print("\nk = ", k, ", calculated based on manhatten distance and majority voting\n\n", a)
        print("\nAccuracy level: ", x)

    else:
        print("Oops. Unexpected parameter. Try again with a valid parameter.")


def main():

    covid = pd.read_csv("covid.csv")

    column_based_train_data = column_based_dataset(covid)
    row_based_test_data1 = [[5, 39.0, 1], [4, 35.0, 0], [3, 38.0, 0], [2, 39.0, 1], [1, 35.0, 0],
                            [0,36.2,0], [5,39.0,1], [2,35.0,0], [3,38.9,1], [0,35.6,0]]

    row_based_test_data2 = [[5, 39.0, 1], [4, 35.0, 0], [3, 38.0, 0],
                           [2, 39.0, 1], [1, 35.0, 0], [0, 36.2, 0],
                           [5, 39.0, 1], [2, 35.0, 0], [3, 38.9, 1],
                           [0, 35.6, 0], [4, 37.0, 0], [4, 36.0, 1],
                           [3, 36.6, 0], [3, 36.6, 1], [4, 36.6, 1]]

    calculate_results_all_k(row_based_test_data1, column_based_train_data)
    calculate_results_all_k(row_based_test_data2, column_based_train_data)

    find_knn_accuracy(row_based_test_data1, column_based_train_data, 3, "m", "majority")
    find_knn_accuracy(row_based_test_data2, column_based_train_data, 15, "e", "distance weight")


if __name__ == "__main__":

    main()


"""
In the real life problems with large training and test sets, we need to choose k value optimal. If k is too small then
noise will have huge influence on the predicted label. However, if we choose k too large, then it becomes computationally 
expensive. We need to choose k which has higher accuracy level with a optimum computation expense. 

Here, we have a small dataset. 

For the first train set:
If we choose k larger than 3 (when k = 2, manhatten distance - majority voting accuracy becomes %80), 
we will have %90 accuracy level. And when we try with different parameters, we never get higher accuracy than %90. However, we 
should not choose k larger than 32. After k=30, accuracy level drops to %40. So the best k value will be between 3 and 32. 

For the second train set:
Accuracy depends on the distance choice and voting choice and most importantly k distance.
However, when we consider both the high accuracy and low computation, k should be between 12 and 19. If we choose k in this 
range, accuracy level will change between 0.73 and 0.8 which is quite good level of accuracy degree. 
"""