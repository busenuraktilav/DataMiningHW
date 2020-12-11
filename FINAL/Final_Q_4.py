import pandas as pd

supp = []


def support(dataset, attr):

    columns = dataset.columns
    sup_attr = 0

    for col in columns:
        sup_attr += dataset[dataset[col] == attr].shape[0]

    return sup_attr


def support_single_calc(dataset, attr_set):
    supp_val = []

    for i in attr_set:
        supp_val.append(support(dataset, i)/len(dataset))

    maxi = max(supp_val)
    index = supp_val.index(maxi)
    best = attr_set[index]

    return best, supp_val


def support_assc(dataset, assc_rule):

    set = dataset

    for rule in assc_rule:
        for i in rule:
            #since item can be any column and there isn't exit a function to find the rows in any column that the item is in, I found the following line from internet for that calculation.
            set = set[set.apply(lambda r: r.str.contains(i, case=False).any(), axis=1)]

    return set.shape[0]

def support_assc_calc(dataset, assc_rules):

    total = dataset.shape[0]
    supp_val_rule = []

    for i in assc_rules:
        supp.append(support_assc(dataset, i))

    for i in supp:
        supp_val_rule.append(float(i)/total)

    best = find_best(supp_val_rule, assc_rules)

    return best, supp_val_rule


def confidence_assc(dataset, assc_rule):

    set = dataset

    if(len(assc_rule) == 1):
        conf = support(dataset, assc_rule[0])

    else:
        for i in assc_rule:
            set = set[set.apply(lambda r: r.str.contains(i, case=False).any(), axis=1)]
        conf = set.shape[0]


    return conf

def confidence_assc_calc(dataset, assc_rules):

    val = []

    for rule in assc_rules:
        val.append(confidence_assc(dataset, rule[0]))

    confidence_assc_val = []

    for i in range(len(val)):
        confidence_assc_val.append(supp[i] / float(val[i]))

    best = find_best(confidence_assc_val, assc_rules)

    return best, confidence_assc_val


def find_best(values, assc_rules):

    maxi = max(values)
    index = values.index(maxi)
    best = assc_rules[index]

    return best

def lift_assc_calc(dataset, assc_rules, confidence):

    val = []

    for rule in assc_rules:
        val.append(support(dataset, rule[1][0])/len(dataset))

    lift_assc_val = []

    for i in range(len(val)):

        if(val[i] == 0):
            lift_assc_val.append(0)
        else:
            lift_assc_val.append(confidence[i] / float(val[i]))

    best = find_best(lift_assc_val, assc_rules)

    return best, lift_assc_val

def main():

    market = pd.read_csv("market_sales.csv")
    attributes = ["whole milk", "yogurt", "coffee", "fruit", "sugar",
                  "hamburger meat", "ketchup", "soda", "chicken", "pork"]

    assc_rules = [[["whole milk"], ["yogurt"]], [["other vegetables"], ["whole milk"]], [["coffee"], ["fruit"]],
                 [["coffee"], ["sugar"]], [["soda"], ["coffee"]], [["hamburger meat"], ["ketchup"]],
                 [["whole milk", "yogurt"], ["coffee"]], [["coffee", "soda"], ["beer"]], [["chicken", "pork"], ["beef"]],
                 [["chicken", "pork", "beef"], ["other vegetables"]]]


    a, support_val = support_single_calc(market, attributes)
    print("\nThe best attribute according to support calculation is ", a, "\nSupport values of attributes are\n", support_val)

    b, support_values = support_assc_calc(market, assc_rules)
    print("\nSupport values for association rules are calculated\n And the best rule is ", b, "\nSupport values for association rules are\n", support_values)

    c, confidence = confidence_assc_calc(market, assc_rules)
    print("\nConfidence calculation for association rules are made\n And the best rule is ", c, "\n", confidence)

    d, lift_values = lift_assc_calc(market, assc_rules, confidence)
    print("\nLift values for the association rules are calculated\n And the best rule is ", d, "\nnLift values for the association rules are\n", lift_values)



if __name__ == "__main__":

    main()