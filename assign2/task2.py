import pandas as pd
import math

heart_data = pd.read_csv("heart_summary.csv")

def gini(dataset, attr_name, attr_val, target):

    class_names = dataset[target].unique()
    elements_in_classes = []
    total = len(dataset.loc[dataset[attr_name] == attr_val])

    for i in class_names:
        d = len(heart_data.loc[(heart_data[attr_name] == attr_val) & (heart_data[target] == i)])
        elements_in_classes.append(d)

    prob_values = []

    for i in range(len(elements_in_classes)):
        prob_values.append(elements_in_classes[i]/float(total))

    gini_val = 1

    for i in prob_values:
        gini_val = gini_val - (i**2)

    return (gini_val)


def entropy(dataset, attr_name, attr_val, target):

    class_names = dataset[target].unique()
    elements_in_classes = []
    total = len(dataset.loc[dataset[attr_name] == attr_val])

    for i in class_names:
        d = len(heart_data.loc[(heart_data[attr_name] == attr_val) & (heart_data[target] == i)])
        elements_in_classes.append(d)

    prob_values = []

    for i in range(len(elements_in_classes)):
        p = elements_in_classes[i]/float(total)

        if(p == 0):
            prob_values.append(0)
        else:
            prob_values.append(p*(math.log(p, 2)))

    entropy_val = 0

    for i in prob_values:
        entropy_val -= i

    return (entropy_val)


def computation_GI_E(dataset, attr_name, target, measure):

    node_names = dataset[attr_name].unique()
    total = len(dataset[target])
    count_node_val = []
    impurity = []

    for i in node_names:

        if(measure == 'gi'):
            g = gini(dataset, attr_name, i, target)
            impurity.append(g)

        elif(measure == 'e'):
            e = entropy(dataset, attr_name, i, target)
            impurity.append(e)

        count_node_val.append(len(dataset.loc[dataset[attr_name] == i]))

    imp_deg = 0

    for i in range(len(impurity)):
        imp_deg += (count_node_val[i]/total)*impurity[i]

    return (imp_deg)


def display(attr_name, entropy, gini):
    print("\nComputation of E and GI for", attr_name, "\n")
    print("Entropy = ", entropy)
    print("Gini Index = ", gini)


#a) computation of E and GI for overall collection of training examples:
total = len(heart_data['target'])
class0 = len(heart_data.loc[heart_data['target'] == 0])
class1 = len(heart_data.loc[heart_data['target'] == 1])
gini_all = 1 - ((class0/float(total))**2) - ((class1/float(total))**2)
entr_all = - ((class0/float(total))*math.log((class0/float(total)), 2)) - ((class1/float(total))*math.log((class1/float(total)), 2))
display('overall collection of training examples', entr_all, gini_all)


#b) computation of E and GI for age:
e_age = computation_GI_E(heart_data, 'age', 'target', 'e')
gi_age = computation_GI_E(heart_data, 'age', 'target', 'gi')
display('age', e_age, gi_age)


#c) computation of E and GI for cp:
e_cp = computation_GI_E(heart_data, 'cp', 'target', 'e')
gi_cp = computation_GI_E(heart_data, 'cp', 'target', 'gi')
display('cp', e_cp, gi_cp)


#d) computation of E and GI for trestbps:
e_trestbps = computation_GI_E(heart_data, 'trestbps', 'target', 'e')
gi_trestbps = computation_GI_E(heart_data, 'trestbps', 'target', 'gi')
display('trestbps', e_trestbps, gi_trestbps)

