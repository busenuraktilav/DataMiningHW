import math

def entropy(num_pos, num_neg):

    p1 = num_pos/(num_pos+num_neg)
    p2 = num_neg/(num_pos+num_neg)

    entr = -p1*(0 if p1==0 else math.log(p1, 2))-p2*(0 if p2==0 else math.log(p2, 2))

    return entr


def entropy_split(num_true_pos, num_true_neg, num_false_pos, num_false_neg):

    pos = num_true_pos+num_false_pos
    neg = num_true_neg+num_false_neg
    t = num_true_pos+num_true_neg
    f = num_false_pos+num_false_neg
    total = t+f

    parent = entropy(pos, neg)
    child1 = entropy(num_true_pos, num_true_neg)
    child2 = entropy(num_false_pos, num_false_neg)

    info_gain = parent - (t/total)*child1 - (f/total)*child2

    return info_gain


def attr_extr(dataset, attr):

    num_true_pos = 0
    num_true_neg = 0
    num_false_pos = 0
    num_false_neg = 0

    j = 0 if attr=="a1" else 1

    for i in range(len(dataset)):

        if(dataset[i][j] == 'T' and dataset[i][2] == '+'):
            num_true_pos += 1

        elif(dataset[i][j] == 'T' and dataset[i][2] == '-'):
            num_true_neg += 1

        elif(dataset[i][j] == 'F' and dataset[i][2] == '+'):
            num_false_pos += 1

        elif(dataset[i][j] == 'F' and dataset[i][2] == '-'):
            num_false_neg += 1

        else:
            print("Oops, something went wrong.")

    return num_true_pos,num_true_neg,num_false_pos,num_false_neg



def main():

    dataset = [['T', 'T', '+'], ['F', 'T', '-'], ['F', 'T', '-'], ['T', 'T', '+'],
               ['T', 'F', '+'], ['F', 'F', '-'], ['T', 'F', '+'], ['F', 'F', '+'],
               ['T', 'F', '+'], ['T', 'F', '-']]

    a,b,c,d = attr_extr(dataset, "a1")
    info_gain_a1 = entropy_split(a, b, c, d)
    a,b,c,d = attr_extr(dataset, "a2")
    info_gain_a2 = entropy_split(a,b,c,d)

    print("\nInformation gain for a1 attribute : ", info_gain_a1)
    print("\nInformation gain for a2 attribute : ", info_gain_a2)


if __name__ == '__main__':

    main()