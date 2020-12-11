import pandas as pd
import itertools
import heapq

#attribute list
def list_attr(dataset):

    attr = list(dataset)
    attr.pop(0)
    attr.pop(-1)

    return attr

#single attribute list -> (Blood_Type), (Can_Fly), ...
def single_attr(attr):

    list_of_attr = []

    for i in attr:
        list_of_attr.append((i,))

    return list_of_attr

#double attribute list -> (Blood_Type, Can_Fly),...
def double_attr(attr):

    combo = []

    for i in range(0, len(attr)):
        for j in range(i+1, len(attr)):
            combo.append((attr[i], attr[j]))

    return combo

#triple attribute list -> (Blood_Type, Can_Fly, Live_Water), ...
def triple_attr(attr):

    combo = []

    for i in range(0, len(attr)):
        for j in range(i+1, len(attr)):
            for k in range(j+1, len(attr)):
                combo.append((attr[i], attr[j], attr[k]))

    return combo

#For each attribute find the attribute value and store it in dictionary -> {"Can_Fly": [yes, no]}
def dict_of_attr(dataset):

    attr = list_attr(dataset)
    dict = {}

    for i in attr:
        values = (dataset[i].unique()).tolist()
        dict[i] = values

    return dict

#for 2 and 3 attributes and its corresponding values to be in the same list
def product_of_tuples(dict, tup):

    arr = []
    for i in tup:
        arr.append(dict[i])

    out = list(itertools.product(*arr))

    return out

#for one attribute and its values to be in the list
def rule_for_one_attr_comb(product, tup, dataset):

    rules = []
    for i in product:
        temp_dataset = dataset
        for j in range(0,len(i)):
            temp_dataset = temp_dataset.loc[temp_dataset[tup[j]] == i[j]]

        if(len(temp_dataset) > 0):
            t = (temp_dataset['Class'].unique()).tolist()

            for k in t:
                arr = []
                for m in range(len(i)):
                    arr.append(tup[m])
                    arr.append(i[m])
                arr.append(k)
                rules.append(arr)

    return rules

#make rules for the given attribute list
def make_rules(dataset, comb_attr):

    dict_attr = dict_of_attr(dataset)

    rules = []
    for i in comb_attr:
        product_tup = product_of_tuples(dict_attr,i)
        temp = rule_for_one_attr_comb(product_tup, i, dataset)
        rules.append(temp)

    return rules

def rule_for_one(dataset, opt):
    attr = list_attr(dataset)

    if(opt == "single"):
        comb = make_rules(dataset, single_attr(attr))

    elif(opt == "double"):
        comb = make_rules(dataset, double_attr(attr))

    elif(opt == "triple"):
        comb = make_rules(dataset, triple_attr(attr))

    else:
        print("invalid option")

    rule_set = []

    for i in comb:
        for j in i:
            rule_set.append(j)

    return rule_set

#make rule for the dataset
def all_rules(dataset):

    attr = list_attr(dataset)

    all = []
    all.append(make_rules(dataset, single_attr(attr)))
    all.append(make_rules(dataset, double_attr(attr)))
    all.append(make_rules(dataset, triple_attr(attr)))

    rule_set = []

    for rule in all:
        for i in rule:
            for j in i:
                rule_set.append(j)

    return rule_set

#calculate coverage rule
def coverage(rule_set, dataset):

    list_coverage = []

    for rule in rule_set: #every rule in ruleset(1-2-3)
        temp_dataset = dataset
        for i in range(0, len(rule)-1, 2):
            temp_dataset = temp_dataset.loc[temp_dataset[rule[i]] == rule[i+1]]

        cov = len(temp_dataset)/len(dataset)
        list_coverage.append(cov)

    return list_coverage

#calculate accuracy rule
def accuracy(rule_set, dataset):

    list_accuracy = []

    for rule in rule_set:
        temp_dataset = dataset
        for i in range(0, len(rule)-1, 2):
            temp_dataset = temp_dataset.loc[temp_dataset[rule[i]] == rule[i+1]]

        acc = len(temp_dataset[temp_dataset['Class'] == rule[-1]])/len(temp_dataset)
        list_accuracy.append(acc)

    return list_accuracy

#Find the rank 10 rule
def ranked_ten(rule_set, arr, k):

    rank = heapq.nlargest(k, arr)
    index = []

    for i in range(len(rank)):
        for j in range(len(arr)):
            if(rank[i] == arr[j]):
                ind = j
                if(ind not in index):
                    index.append(ind)
                    break

    rules = []

    for i in index:
        rules.append(rule_set[i])

    return rules, rank, index


def print_all_rules(rule_set, rank=None, var=None, index=None):

    if(rank):

        count = 0
        print("\n------------------ Rank Rules According to ", var, " (TOP", len(rank), "RANK) ------------------\n")

        for rule in rule_set:

            if(len(rule) == 3):
                print("Rule", index[count], " ----- (", rule[0], "=", rule[1], "->", rule[2], ") ----- ", var, " : ", rank[count]*100, "%")

            elif(len(rule) == 5):
                print("Rule", index[count], " ----- (", rule[0], "=", rule[1], "^", rule[2], "=", rule[3], "->", rule[4], ") ----- ", var, " : ", rank[count]*100, "%")

            elif(len(rule) == 7):
                print("Rule", index[count], " ----- (", rule[0], "=", rule[1], "^", rule[2], "=", rule[3], "^", rule[4], "=", rule[5], "->", rule[6], ") ----- ", var, " : ", rank[count]*100, "%")

            else:
                print("Oops, something went wrong")

            count+=1

    else:

        print("Total Rule Count: ", len(rule_set), "\n")

        for rule in range(len(rule_set)):

            if(len(rule_set[rule]) == 3):
                print("Rule", rule, " ----- ", rule_set[rule][0], "=", rule_set[rule][1], "->", rule_set[rule][2])

            elif(len(rule_set[rule]) == 5):
                print("Rule", rule, " ----- ", rule_set[rule][0], "=", rule_set[rule][1], "^", rule_set[rule][2], "=", rule_set[rule][3], "->", rule_set[rule][4])

            elif(len(rule_set[rule]) == 7):
                print("Rule", rule, " ----- ", rule_set[rule][0], "=", rule_set[rule][1], "^", rule_set[rule][2], "=", rule_set[rule][3], "^", rule_set[rule][4], "=", rule_set[rule][5], "->", rule_set[rule][6])

            else:
                print("Oops, something went wrong")


def main():

    vertebrates = pd.read_csv("vertebrates.csv")

    print("Number of Classes : ", len(vertebrates['Class'].unique()))
    print("Number of Attributes : ", len(vertebrates.columns)-2)

    single = rule_for_one(vertebrates, "single")
    double = rule_for_one(vertebrates, "double")
    triple = rule_for_one(vertebrates, "triple")

    print("\n------------------ Single Rules ------------------\n")
    print_all_rules(single)
    print("\n------------------ Double Rules ------------------\n")
    print_all_rules(double)
    print("\n------------------ Triple Rules ------------------\n")
    print_all_rules(triple)


    rule_set = all_rules(vertebrates)
    cov = coverage(rule_set, vertebrates)
    cov_ten, cov_rank, cov_index = ranked_ten(rule_set, cov, 10)
    acc = accuracy(rule_set, vertebrates)
    acc_ten, acc_rank, acc_index = ranked_ten(rule_set, acc, 10)

    print("\n------------------ All Rules ------------------\n")
    print_all_rules(rule_set)

    print_all_rules(cov_ten, cov_rank, "Coverage", cov_index)
    print_all_rules(acc_ten, acc_rank, "Accuracy", acc_index)


if __name__ == '__main__':

    main()
