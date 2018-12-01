import numpy as np
import sys
import math

#Splitting the dataset into left and right
def split_dataset(value, j, dataset):
    left = list()
    right = list()
    for i in range(dataset.shape[0]):
        if j not in categorical_attr and dataset[i][j] <= value:
                left.append(dataset[i])
        elif dataset[i][j] == value:
                left.append(dataset[i])
        else:
                right.append(dataset[i])
    return np.array(left), np.array(right)


#Calculating the gini index
def gini_calculation(left, right):
    left_zeros = left_ones = right_zeros = right_ones = 0
    if len(left):
        left_zeros =  float(len(list(filter(None, [e == 0 for e in list(left[:,-1]) ]))))/len(left)
        left_ones = float(len(list(filter(None, [e == 1 for e in list(left[:,-1]) ]))))/len(left)
    if len(right):
        right_zeros =  float(len(list(filter(None, [e == 0 for e in list(right[:,-1]) ]))))/len(right)
        right_ones =  float(len(list(filter(None, [e == 1 for e in list(right[:,-1]) ]))))/len(right)
    gini_left = (1.0 - ((left_ones **2) + (left_zeros **2)))*len(left)
    gini_right = (1.0 - ((right_ones **2) + (right_zeros **2)))*len(right)
    gini = (gini_left  + gini_right) / (len(left)+len(right))
    return gini


#Classifying at a particular node
def classify_node(left, right):
    zeros = ones = 0
    if len(left) :
        zeros += len(list(filter(None, [e == 0 for e in list(left[:,-1])])))
        ones += len(list(filter(None, [e == 1 for e in list(left[:,-1])])))
    if len(right) :
        zeros += len(list(filter(None, [e == 0 for e in list(right[:,-1])])))
        ones += len(list(filter(None, [e == 1 for e in list(right[:,-1])])))
    if ones > zeros :
        return 1
    else :
        return 0


#Determining the split point based on the gini index
def get_split_point(dataset):
    minimum_error = sys.float_info.max
    for j in range(dataset.shape[1] - 1):
        for i in range(dataset.shape[0]):
            left, right = split_dataset(dataset[i][j], j,dataset)
            gini = gini_calculation(left, right)
            if gini < minimum_error:
                minimum_error = gini
                point={'id':i,'value':dataset[i][j],'attr':j,'left':left,'right':right}
    return point


#Recursive function to build the decision tree
def build_decision_tree(node, depth):
    left = node['left']
    node.pop('left',None)
    right = node['right']
    node.pop('right',None)
    if len(left) == 0 or len(right) == 0:
        node['left'] = node['right'] = classify_node(left, right)
        return node
    if len(list(set(left[:,-1]))) == 1:
        node['left'] = classify_node(left, [])
    else:
        node['left'] = build_decision_tree(get_split_point(left), depth+1)
    if len(list(set(right[:,-1]))) == 1:
        node['right'] = classify_node([], right)
    else:
        node['right'] = build_decision_tree(get_split_point(right), depth+1)
    return node


#Predicting the test data using the node of decision tree
def prediction(test_row, node):
    if test_row[node['attr']] < node['value']:
        if type(node['left']) is not dict:
            return node['left']
        else:
            return prediction(test_row, node['left'])
    else:
        if type(node['right']) is not dict:
            return node['right']
        else:
            return prediction(test_row, node['right'])


#Calculation of the true positive, true negative, false positive and false negative
def calc_measures(actual_class, predicted_class):
    tp = tn = fp = fn = 0
    for i in range(len(actual_class)):
        if actual_class[i]==1 and predicted_class[i]==1:
            tp=tp+1
        elif actual_class[i]==0 and predicted_class[i]==0:
            tn=tn+1
        elif actual_class[i]==0 and predicted_class[i]==1:
            fp=fp+1
        elif actual_class[i]==1 and predicted_class[i]==0:
            fn=fn+1
    return tp, tn, fp, fn


#Input of the text file(dataset) and the number of learners from the user
filename = input("Enter the filename: ")
no_learners = int(input("Enter the number of learners: "))
with open(filename) as textFile:
    lines=[line.split() for line in textFile]
input_data = np.array(lines)
class_labels = np.array(input_data[:,-1].reshape((len(input_data),1)),dtype=int)
attributes = input_data[:,0:-1]


#Converting categorical attributes into numerical attributes
categorical_attr = list()
new_attr= list()
for i in range(attributes.shape[1]):
    try:
        float(attributes[0][i])
    except ValueError:
        categorical_attr.append(i)
for i in categorical_attr:
    unique_attr = list(set(attributes[:,i]))
    new_attr.extend(range(len(unique_attr)))
    mapper = dict(map(lambda x,y:(x,y),unique_attr,new_attr))
    for j in range(len(attributes[:,i])):
        attributes[j][i] = mapper[attributes[j][i]]


#Splitting the data into 10 data chunks for cross validation
accuracy = precision = recall = f1_score = 0.0
attributes=np.array(attributes, dtype=float)
data=np.concatenate((attributes,class_labels),axis=1)
data_chunks = np.array_split(data,10)


#Iterating the algorithm for 10 folds
for idx in range(10):
    train_data = np.array(np.concatenate([y for (x,y) in enumerate(data_chunks,0) if x!=idx],axis=0))
    len_train = len(train_data)
    test_data = np.array(data_chunks[idx])
    test_class_all = list()
    test_class_learner = list()

    initial_weight = [1.0/len(train_data)] * len(train_data)
    weights = np.asarray(initial_weight)
    error = 0.0
    j = 0
    while j < no_learners:
        test_class = list()
        train_class = list()
        error_list = list()
        root={}
        select = np.random.choice(len_train, len_train, replace=True, p = weights)
        root = get_split_point(train_data[select,:])
        root = build_decision_tree(root, 1)
        for i in range(len(train_data)):
            train_class.append(prediction(train_data[i],root))
        original=train_data[:,-1]
        predicted=np.asarray(train_class)
        for i in range(len(predicted)):
            if predicted[i] == original[i]:
                error_list.append(-1)
            else:
                error_list.append(1)
        error_list=np.asarray(error_list)

        #Calculating the error for the classifier
        nr = np.sum(np.multiply(weights, np.absolute(predicted - original)))
        dr = np.sum(weights)
        error = nr/dr
        #Rejecting the learner if error is greater than 0.5
        if error < 0.5:
            # Calculating the importance
            imp = 0.5 * math.log((1-error)/error)
            #Updating the weight of each record based on classification error
            weights = np.multiply(weights, np.exp(imp * error_list))
            weights = weights / np.sum(weights)

            #implementng it on the test data
            for i in range(len(test_data)):
                if(prediction(test_data[i],root) == 1):
                    test_class.append(1)
                else:
                    test_class.append(-1)
            test_class = np.asarray(test_class).astype(float)
            test_class_learner.append(test_class * imp)
            j = j+1
    #Combining the results of all the learners
    test_class_all = np.sum(test_class_learner, axis=0)
    for i in range(len(test_class_all)):
        if(test_class_all[i] > 0.0):
            test_class_all[i] = 1
        else:
            test_class_all[i] = 0

    actual_class = test_data[:,-1]
    tp, tn, fp, fn = calc_measures(actual_class, test_class_all)
    accuracy += (float((tp + tn)/(tp + fn + fp + tn)))*10
    if (tp+fp)!=0:
        precision += (float((tp)/(tp + fp)))*10
    if(tp+fn)!=0:
        recall += (float((tp)/(tp + fn)))*10
f1_score = 0.01*2*(precision*recall)/(precision+recall)
print("Accuracy: "+str(accuracy))
print("Precision: "+str(precision))
print("Recall: "+str(recall))
print("F1 Measure: "+str(f1_score))
