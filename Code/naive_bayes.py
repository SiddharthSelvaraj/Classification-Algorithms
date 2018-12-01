import numpy as np
from sklearn import preprocessing
from random import seed
from random import randrange
import numpy_indexed as npi
from heapq import heappush,heappop
#from sklearn.model_selection  import KFold
import math
np.set_printoptions(threshold=np.nan)

categorical_data = []

def preprocessInput(file):
	
	data = np.genfromtxt(file, delimiter = '\t')
	features = np.genfromtxt(file, usecols = range(0, data.shape[1]-1),dtype='str')
	class_label = np.genfromtxt(file, usecols = data.shape[1]-1,dtype='str')
	
	print("features")
	#print(features[1])
	#print(class_label)
	
	#References: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.unique.html
	for i in range(0,features.shape[1]):
		try:
			features[:,i] = features[:,i].astype(np.float)
		except:
			ft_array,indices = np.unique(features[:,i],return_inverse = True)
			print(indices)
			features[:,i] = indices.astype(np.float)
			#print('ft_array')
			#print(features[:,i])
			#print('indices')
			categorical_data.append(i)
	features = features.astype(np.float)
	class_label = class_label.astype(np.float)
	

	return features, class_label

#Function to split dataset into train and test - 10 fold cross-validation
def split(index,data_chunks,labels):
	train_data = np.asarray(np.vstack([x for i,x in enumerate(data_chunks) if i != index]))
	train_labels = np.asarray(np.concatenate([x for i,x in enumerate(labels) if i != index]))
	
	test_data = np.asarray(data_chunks[index])
	test_labels = np.asarray(labels[index])
	
	return train_data, train_labels, test_data, test_labels
	
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
	
def computeProbability(train_cont_data, test_cont_data):
	tmean = np.mean(train_cont_data,axis=0)
	std_dev = np.std(train_cont_data, axis=0)
	num = np.power(test_cont_data-tmean,2)
	denom = 2*np.power(std_dev,2)
	exponent = np.exp(-num/denom)
	return (1/(np.sqrt(math.pi*2)*std_dev)) * exponent

#def cross_validation_for_demo(features,label):
	
	
def cross_validation(features, label):
	temp_features = np.array_split(features,10)
	temp_label = np.array_split(label,10)
	avg_accuracy,avg_precision,avg_recall,avg_fMeasure = 0,0,0,0
	for ind in range(10):
		train_data, train_labels, test_data, test_labels = split(ind,temp_features,temp_label)
		print(test_data)
		cat_prob_zero = [1 for i in range(0, len(test_data))]
		cat_prob_one = [1 for i in range(0,len(test_data))]
		train_continuous_data = np.delete(train_data,categorical_data,axis=1)
		test_continuous_data = np.delete(test_data,categorical_data,axis=1)
		#print("Train Num data:")
		#print(np.mean(train_continuous_data,axis=0))
		
		train_cont_zero = train_continuous_data[train_labels == 0.0]
		
		train_cont_one = train_continuous_data[train_labels == 1.0]
		#print(train_continuous_data)
		continuous_prob_zero = computeProbability(train_cont_zero, test_continuous_data)
		continuous_prob_one = computeProbability(train_cont_one, test_continuous_data)
		
		cont_prob_zero_list = list()
		#cont_prob_zero = np.asarray(cont_prob_zero_list)
		cont_prob_one_list = list()
		#cont_prob_one = np.asarray(cont_prob_one_list)
		
		for i in range(len(test_data)):
			cont_prob_zero_list.append(np.prod(continuous_prob_zero[i]))
			cont_prob_one_list.append(np.prod(continuous_prob_one[i]))
		
		cont_prob_zero = np.asarray(cont_prob_zero_list)
		cont_prob_one = np.asarray(cont_prob_one_list)
		
		print("continuous __prob zero")
		print(cont_prob_zero_list)
		test_predicted_labels = list()
		
		if len(categorical_data) != 0:
			
			train_categorical_data = (train_data[:,categorical_data]).ravel()
			test_categorical_data = (test_data[:,categorical_data]).ravel()
			#print("Test categorical data")
			#print(test_categorical_data)
			#print("Train categorical Data")
			#print(train_categorical_data)
			for i in range(len(test_categorical_data)):
				
				count_zero = list(train_labels).count(0.0)
				
				prior_zero = count_zero/len(train_labels)
				
				pOfX = 1
				prob = dict()
				
				countCat = np.sum((train_categorical_data==test_categorical_data[i]))
				pOfX *= countCat/len(train_labels)
				#print("probab val is")
				#print(pOfX)
				desc_post_prob_zero = 1
				
				class_zero = 0
				for j in range(0,len(train_categorical_data)):
					if(train_categorical_data[j] == test_categorical_data[i]):
						if(train_labels[j]==0.0):
							class_zero+=1
				#print("class_zero val is")
				#print(class_zero)
				desc_post_prob_zero *= class_zero/count_zero
				prob[0] = (desc_post_prob_zero*prior_zero)/pOfX
				#print("prb0 val is")
				#print(prob[0])
				
				count_one = list(train_labels).count(1.0)
				prior_one = count_one/len(train_labels)
				class_one = 0
				desc_post_prob_one = 1
				for j in range(0,len(train_categorical_data)):
					if(train_categorical_data[j] == test_categorical_data[i]):
						if(train_labels[j]==1.0):
							class_one+=1
				#print("class_one val is")
				#print(class_one)
				desc_post_prob_one *= class_one/count_one
				prob[1] = (desc_post_prob_one*prior_one)/pOfX
				#print("prb1 val is")
				#print(prob[1])
				#print("Prob 0 + prob 1")
				#print(prob[0]+prob[1])
				cat_prob_zero.append(prob[0])
				cat_prob_one.append(prob[1])
		
		cat_prob_zero = np.array(cat_prob_zero)
		cat_prob_one = np.array(cat_prob_one)
		
		#print("catgorical prob zero")
		#print(len(cat_prob_zero))
		
		#cont_prob_zero = np.prod(continuous_prob_zero)
		#print(continuous_prob_zero)
		for i in range(0,len(test_data)):
			class_prob_zero = cont_prob_zero[i]* cat_prob_zero[i]
			class_prob_one = cont_prob_one[i]* cat_prob_one[i]
			#print("Printing class zero")
			#print(class_prob_zero)
			if(class_prob_one>class_prob_zero):
				test_predicted_labels.append(1.0)
			else:
				test_predicted_labels.append(0.0)
		#print("Testing class")
		#print(test_predicted_labels)
		test_predicted_labels = np.asarray(test_predicted_labels)
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
		
		
file = input("Enter file name: ")
features, labels = preprocessInput(file)
'''print(features)
features = np.array_split(features,10)

labels = np.array_split(labels,10)
for ind in range(10):
	train_data, train_labels, test_data, test_labels = split(ind,features,labels)
	print(train_data)'''
cross_validation(features,labels)	
#cross_validation_for_demo(features,labels)