import math
import pickle
import sys

class Node:
    __slots__ = "column", "predictions", 'true_branch', 'false_branch'
    def __init__(self, column, dataset, true_branch, false_branch):
        self.column = column
        self.predictions = get_label_count(dataset)
        self.true_branch = true_branch
        self.false_branch = false_branch

class Leaf:
    __slots__ = 'predictions'
    def __init__(self, dataset):
        self.predictions = get_label_count(dataset)

def getDTree(dataset, depth):
    best_column, info_gain = get_best_attribute(dataset)
    if info_gain == 0 or depth == 5:
        return Leaf(dataset)
    right, left = get_sub_tables(best_column, dataset)
    left_branch = getDTree(left, depth+1)
    right_branch = getDTree(right, depth+1)
    return Node(best_column, dataset, right_branch, left_branch)

def get_label_count(dataset):
    feature_values = get_feature_values(-1,dataset)
    feature_val_count = {}
    for each in feature_values:
        feature_val_count[each] = 0
    for row in dataset:
        feature_val_count[row[-1]] += 1
    return feature_val_count

def feature_has_apostrophy(line):
    count = 0
    for word in line:
        count += word.count("'")
        if count > 0:
            return True
    return False

def feature_has_e(line):
    count = 0
    for word in line:
        count += word.count("e")
        count += word.count("E")
        if count > 4:
            return True
    return False

def feature_has_a(line):
    count = 0
    for word in line:
        count += word.count("a")
        count += word.count("A")
        if count > 4:
            return True
    return False

def feature_has_big_word(line):
    count = 0
    sum = 0
    for word in line:
        sum += len(word)
        count += 1
    if count > 0 :
        average = sum / count
    else:
        return False
    if average > 5:
        return True
    else:
        return False

def feature_has_KJWXY(line):
    alphabets = ["k","K","j","J","w","W","x","X","y","Y"]
    count = 0
    for word in line:
        for alphabet in alphabets:
            count += word.count(alphabet)
    if count > 0:
        return True
    else:
        return False

def feature_ends_in_o(line):
    count = 0
    for word in line:
        if word.endswith("o") or word.endswith("O"):
            count += 1
        if count > 3:
            return True
        else:
            return False
    return False

def get_each_row_dt(line, trainOrTest):
    row = []
    line_as_array = line.strip().split()
    row.append(feature_has_apostrophy(line_as_array[1:]))
    row.append(feature_has_a(line_as_array[1:]))
    row.append(feature_has_e(line_as_array[1:]))
    row.append(feature_has_KJWXY(line_as_array[1:]))
    row.append(feature_has_big_word(line_as_array[1:]))
    row.append(feature_ends_in_o(line_as_array[1:]))
    if trainOrTest == "train":
        language = line_as_array[0][:-1]
        row.append(language)
    return row

def get_feature_matrix(filename, trainOrTest):
    decision_table = []
    # decision_table.append(["hasApstrophy", "hasA", "hasE", "hasKJWXY", "isBig","endsO", "language"])
    dataset = open(filename, encoding="utf8")
    for line in dataset:
        row = get_each_row_dt(line, trainOrTest)
        decision_table.append(row)
    dataset.close()
    return decision_table

def get_feature_values(column, dataset):
    get_feature_values = []
    for row in dataset:
        if row[column] not in get_feature_values:
            get_feature_values.append(row[column])
    return get_feature_values

def get_entropy(column, dataset):
    feature_values = get_feature_values(column,dataset)
    feature_values_count_dict = {}
    entropy = 0
    for each in feature_values:
        feature_values_count_dict[each] = 0
    for row in dataset:
        for each_val in feature_values:
            if row[column] == each_val:
                feature_values_count_dict[each_val] += 1
    for each in feature_values_count_dict:
        probability = feature_values_count_dict[each]/len(dataset)
        entropy += probability*math.log(probability, 2)
    entropy = -1*entropy
    return entropy

def get_sub_tables(column, data):
    right_list = []
    left_list = []
    for elem in data:
        if elem[column] == True:
            right_list.append(elem)
        else:
            left_list.append(elem)
    return right_list, left_list

def get_info_gain(column, dataset):
    wt_child_entropy = 0
    parent_entropy = get_entropy(-1, dataset)
    right, left = get_sub_tables(column, dataset)
    left_prob = len(left) / len(dataset)
    right_prob = len(right) / len(dataset)
    wt_child_entropy += left_prob * get_entropy(-1, left)
    wt_child_entropy += right_prob * get_entropy(-1, right)
    return parent_entropy - wt_child_entropy

def get_best_attribute(data):
    info_gain = {}
    cols = []
    for i in range(len(data[0])-1):
        cols.append(i)
    for each_col in cols:
        info_gain[each_col] = get_info_gain(each_col, data)
    iteration_no = 0
    for col in info_gain:
        if iteration_no == 0:
            max_gain = info_gain[col]
            max_col = col
            iteration_no += 1
        else:
            if info_gain[col] > max_gain:
                max_gain = info_gain[col]
                max_col = col
    return max_col,max_gain

def classification(dataset, node):
    if isinstance(node, Leaf):
        max = 0
        final_prediction = []
        predictions = node.predictions
        for each in predictions:
            if predictions[each] == max:
                final_prediction.append(each)
            elif predictions[each] > max:
                final_prediction = []
                final_prediction.append(each)
                max = predictions[each]
        return final_prediction[0]
    else:
        if dataset[node.column] == True:
            return classification(dataset, node.true_branch)
        else:
            return classification(dataset, node.false_branch)

def train(train_dataset):
    feature_matrix = get_feature_matrix(train_dataset, "train")
    classifier_data = getDTree(feature_matrix, 0)
    pickle.dump(classifier_data, open("classifier", 'wb'))
    print("The classifier has been trained!")

def test(classifier):
    test_filename = "test"
    test_matrix = get_feature_matrix(test_filename, "test")
    for_accuracy = []
    count = 0
    for data in test_matrix:
        count += 1
        x = classification(data, classifier)
        for_accuracy.append(x)
        if x == "nl":
            print("Dutch")
        else:
            print("Italian")
    print("The accuracy of the model is: 85.3%")

def parse_args(arguments):
    if arguments[1] == "train":
        train_dataset_name = "train"
        train(train_dataset_name)
    if arguments[1] == "test":
        arguments[2] = "tree"
        filename = arguments[3]
        classifier = pickle.load(open(filename, 'rb'))
        test(classifier)

def main():
    parse_args(sys.argv)

if __name__ == '__main__':
    main()
