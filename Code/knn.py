import numpy as np
from sklearn import preprocessing
from random import seed
from random import randrange
import numpy_indexed as npi
from heapq import heappush,heappop
#from sklearn.model_selection  import KFold
np.set_printoptions(threshold=np.nan)

categorical_data = []

def preprocessInput(file):
	
	data = np.genfromtxt(file, delimiter = '\t')
	features = np.genfromtxt(file, usecols = range(0, data.shape[1]-2),dtype='str')
	class_label = np.genfromtxt(file, usecols = data.shape[1]-1,dtype='str')
	
	print("features")
	#print(features[1])
	#print(class_label)
	
	for i in range(0,features.shape[1]):
		try:
			features[:,i] = features[:,i].astype(np.float)
		except:
			ft_array,indices = np.unique(features[:,i],return_inverse = True)
			features[:,i] = indices.astype(np.float)
			categorical_data.append(i)
	features = features.astype(np.float)
	class_label = class_label.astype(np.float)
	#print(features)

	return features, class_label

def euclidean_dist(x, y):
	distance = 0
	for i in range(len(x)):
		if(i not in categorical_data):
			distance += np.square(x[i]-y[i])
		else:
			if(x[i]!=y[i]):
				distance += 1
	return np.sqrt(distance)

#Function to split dataset into train and test - 10 fold cross-validation
def split(index,data_chunks,labels):
	train_data = np.asarray(np.vstack([x for i,x in enumerate(data_chunks) if i != index]))
	train_labels = np.asarray(np.concatenate([x for i,x in enumerate(labels) if i != index]))
	
	test_data = np.asarray(data_chunks[index])
	test_labels = np.asarray(labels[index])
	
	return train_data, train_labels, test_data, test_labels

def split_for_demo(train_file,test_file):
	train_data = np.genfromtxt(train_file, delimiter = '\t')
	train_features = np.genfromtxt(train_file, usecols = range(0, train_data.shape[1]-2),dtype='str')
	train_class_label = np.genfromtxt(train_file, usecols = train_data.shape[1]-1,dtype='str')
	
	test_data = np.genfromtxt(test_file, delimiter = '\t')
	test_features = np.genfromtxt(test_file, usecols = range(0, test_data.shape[1]-2),dtype='str')
	test_class_label = np.genfromtxt(test_file, usecols = test_data.shape[1]-1,dtype='str')
	
	train_features = train_features.astype(np.float)
	train_class_label = train_class_label.astype(np.float)
	test_features = test_features.astype(np.float)
	test_class_label = test_class_label.astype(np.float)
	
	return train_features, train_class_label, test_features, test_class_label
	
	
def cross_validation_split(dataset, folds=10):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

'''def neighbors(train_data, test_data, index, folds):
	heap = list()
	neighbors = list()
	for i in range(len(train_data)):
		distance = euclidean_dist(train_data[i],test_data[index])
		heappush
'''	
def kNN(train_data,train_labels, test_data, neighbor_val):
	test_labels = list()
	for index in range(len(test_data)):
		result = list()
		neighbors = list()
		labels = list()
		for i in range(len(train_data)):
			distance = euclidean_dist(train_data[i], test_data[index])
			heappush(result, (distance, i))
		for i in range(neighbor_val):
			neighbors.append(heappop(result))
		for neighbor in neighbors:
			label_val = train_labels[neighbor[-1]]		
			labels.append(label_val)
		test_labels.append(max(set(labels),key=labels.count))
	
	return test_labels		
	
def evaluate_performance(actual_test_label, predicted_test_label):
	a = b = c = d = 0
	accuracy = precision = recall = f1_measure = 0 
	for i in range(len(predicted_test_label)):
		if actual_test_label[i] == 1 and predicted_test_label[i] == 1:
			a += 1
		elif actual_test_label[i] == 1 and predicted_test_label[i] == 0:
			b += 1
		elif actual_test_label[i] == 0 and predicted_test_label[i] == 1:
			c += 1
		elif actual_test_label[i] == 0 and predicted_test_label[i] == 0:
			d += 1
	accuracy += (float(a+d)/(a+b+c+d))
	if(a+c != 0):
		precision += (float(a)/(a+c))
	if(a+b != 0):
		recall += (float(a)/(a+b))
	f1_measure += (float(2*a)/((2*a)+b+c))
	return accuracy, precision, recall, f1_measure
	
def cross_validation(features, label, folds):
	temp_features = np.array_split(features,10)
	temp_label = np.array_split(label,10)
	avg_accuracy,avg_precision,avg_recall,avg_fMeasure = 0,0,0,0
	for ind in range(10):
		train_data, train_labels, test_data, test_labels = split(ind,temp_features,temp_label)
		test_predicted_labels = kNN(train_data, train_labels, test_data, folds)
		print("Printing test_predicted_labels")
		test_predicted_labels = np.asarray(test_predicted_labels)
		print(test_predicted_labels)
		print("Printing test actual labels")
		print(test_labels)
		accuracy, precision, recall, f1_measure = evaluate_performance(test_labels, test_predicted_labels)
		print("accuracy:",accuracy*10)
		avg_accuracy += accuracy*10
		print("precision:",precision*10)
		avg_precision += precision*10
		print("recall:",recall*10)
		avg_recall += recall*10
		print("f1_measure:",f1_measure)
		avg_fMeasure += f1_measure*0.1
	print("Average Accuracy:",avg_accuracy)
	print("Average Precision:",avg_precision)
	print("Average Recall:",avg_recall)
	print("Average F_measure:",avg_fMeasure)
		
	
	
'''
file = input("Enter file name: ")
features, labels = preprocessInput(file)
k = int(input("Enter the value for k: "))
cross_validation(features,labels,k)

'''
train_file = input("Enter train file name: ")
test_file = input("Enter test fileName: ")
k = int(input("Enter the value for k: "))
train_data, train_labels, test_data, test_labels = split_for_demo(train_file,test_file)
test_predicted_labels = kNN(train_data, train_labels, test_data, k)
print("Printing test_predicted_labels")
test_predicted_labels = np.asarray(test_predicted_labels)
print(test_predicted_labels)
print("Printing test actual labels")
print(test_labels)
accuracy, precision, recall, f1_measure = evaluate_performance(test_labels, test_predicted_labels)
print("accuracy:",accuracy*100)
print("precision:",precision*100)
print("recall:",recall*100)
print("f1_measure:",f1_measure)


