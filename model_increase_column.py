import csv
import numpy as np
import sys
import getopt
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from pyod.models.cof import COF
from pyod.models.loci import LOCI
import math

def writeLabelScore(label,score):
	with open("trainingData.csv") as csvfile:
		csv_reader = csv.reader(csvfile)
		header = next(csv_reader)
		header.append('label')
		header.append('score')
		rows = []
		i = 0
		for row in csv_reader:
			row.append(label[i])
			row.append(score[i])
			i = i + 1
			rows.append(row)
		csvfile.close()
	#print rows
	with open("labelData_increase_column.csv","wb") as csvfile:
		csv_writer = csv.writer(csvfile)
		csv_writer.writerow(header)
		csv_writer.writerows(rows)
		csvfile.close()
	return
def writeLabel(label):
	with open("trainingData.csv") as csvfile:
		csv_reader = csv.reader(csvfile)
		header = next(csv_reader)
		header.append('label')
		rows = []
		i = 0
		for row in csv_reader:
			row.append(label[i])
			i = i + 1
			rows.append(row)
		csvfile.close()
	#print rows
	with open("labelDatav2.csv","wb") as csvfile:
		csv_writer = csv.writer(csvfile)
		csv_writer.writerow(header)
		csv_writer.writerows(rows)
		csvfile.close()
	return


# function : help
def help():
	print ('The script provides several outlier detection algorithms with already-tuned parameter to use.')
	print ('Algorithms and related options:')
	print ('LOCI : -o or --loci')
	print ('COF : -c or --cof')
	print ('DBSCAN : -d or --dbscan')
	print ('LOF: -l or --lof')
	print ('iForest: -i or --iforest')


# function : loci
# parameter needed to be tuned: alpha, k
def loci(X):
	alpha = 0.5
	k = 3
	clf = LOCI(alpha=alpha, k=k)
	clf.fit(X)
	label = clf.labels_
	#return label
	writeLabel(label)
	return


# function : cof
# parameter needed to be tuned: contamination, n_neighbors
def cof(X):
	contamination_factor = 0.1
	k = 20
	clf = COF(contamination=contamination_factor,n_neighbors=k)
	clf.fit(X)
	label = clf.labels_
	score = clf.decision_scores_
	threshold = clf.threshold_
	writeLabel(label)
	return


# function: dbscan
# parameter needed to be tuned : eps, min_sample
def dbscan(X):
	epsilon = 0.2
	min_sample_num = 4
	clf = DBSCAN(eps=epsilon,
				metric="euclidean",
				min_samples=4,
				n_jobs=1)
	label = clf.fit_predict(X)
	i = 0
	while i < len(label):
		if label[i] != -1:
			label[i] = 0
		else:
			label[i] = 1
		i = i + 1
	writeLabel(label)
	return


# function : lof
# parameter needed to be tuned: k
def lof(X):
	k = 20 # k can be tuned
	clf = LocalOutlierFactor(n_neighbors=k)
	label = clf.fit_predict(X)
	# score = clf.negative_outlier_factor_;
	i = 0
	while i < len(label):
		if label[i] != -1:
			label[i] = 0
		else:
			label[i] = 1
		i = i + 1
	writeLabel(label)
	return



# function : iForest
# parameter needed to be tuned: n_estimators, max_sample, contamination
def iForest(X):
	rng = np.random.RandomState(6324)
	num_estimators = 50
	max_sample_num = 256
	clf = IsolationForest(n_estimators=num_estimators,n_jobs=1,behaviour='new', max_samples=max_sample_num, random_state=rng, contamination='auto')
	clf.fit(X)
	label = clf.predict(X)
	scores = clf.score_samples(X)
	i = 0
	while i < len(label):
		if label[i] != -1:
			label[i] = 0
		else:
			label[i] = 1
		i = i + 1
	writeLabelScore(label,scores)
	return

def extractScale(column_data):
    res = np.zeros(column_data.shape)
    for i in range(column_data.shape[0]):
        scale=0
        while(math.fabs(column_data[i])<1):
            column_data[i] *= 10
            scale = scale + 1
        res[i]=scale
    return res

def dataExtend(orig_data,pre_data):
    dim0 = orig_data.shape[0]
    dim1 = (orig_data.shape[1])*2 - 1
    extend_data = np.zeros((dim0,dim1))
    for i in range(dim1):
        if(i%2==0):
            extend_data[:,i] = pre_data[:,i/2]
        else:
            extend_data[:,i] = extractScale(extend_data[:,i-1])
    return extend_data



#print X_training_scaled

# read opts and call related function
try:
	opts, args = getopt.getopt(sys.argv[1:],'ildcoh',['iforest','lof','dbscan','cof','loci','help'])
except getopt.GetoptError as e:
	print ('Getopt error and exit, error info:%s' % str(e))
if opts == []:
		help()
else:
	# open training data file
	# Read data into X_training array
	header = []
	dataList = []
	with open("trainingData.csv") as csvfile:
		csv_reader = csv.reader(csvfile)
		header = next(csv_reader)
		for row in csv_reader:
			temp = []
			i = 5
			while (i < len(row)):
				temp.append(float(row[i]))
				i = i + 1
			dataList.append(temp)
		csvfile.close()

	# original data
	X_training = np.array(dataList)

	# z-score normalized training data
	X_training_scaled = preprocessing.scale(X_training)
	X_extended = dataExtend(X_training,X_training_scaled)
	assert(X_extended.shape[1]==2*X_training.shape[1]-1)
    
	for opt,arg in opts:
		if opt in ("-i","--iforest"):
			iForest(X_extended)
		elif opt in ("-l","--lof"):
			lof(X_training_scaled)
		elif opt in ("-d","--dbscan"):
			dbscan(X_training_scaled)
		elif opt in ("-c","--cof"):
			cof(X_training_scaled)
		elif opt in ("-o","--loci"):
			loci(X_training_scaled)
		elif opt in ("-h","--help"):
			help()

#print label