import csv
import numpy as np
import sys
import getopt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from pyod.models.cof import COF
from pyod.models.loci import LOCI

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
	with open("labelDatav2.csv","wb") as csvfile:
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
	with open("labelData_dbscan.csv","wb") as csvfile:
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
    eps_range = np.arange(0.1,0.3,0.02)
    min_samples_range = np.arange(2,20,2)
    permutation = np.random.permutation(X.shape[0])
    X_shuffled = X[permutation,:]
    X_shuffled = X_shuffled[60000:100001]
    silhouette_max = 0
    max_eps = 0
    max_minSamples = 0
    scores_data = []
    for eps in eps_range:
        for min_samples in min_samples_range:
            try:
                model = DBSCAN(eps=eps,min_samples=min_samples).fit(X_shuffled)
                labels = model.labels_
                s_score = metrics.silhouette_score(X_shuffled,labels)
                print("Eps = " + str(eps) + "; min_samples = " + str(min_samples)+"; score = "+str(s_score))
                score = (eps,min_samples,s_score)
                scores_data.append(score)
                if s_score>silhouette_max:
                    silhouette_max = s_score
                    max_eps = eps
                    max_minSamples = min_samples
            except:
                model=''
            else:
                model=''
    i = 0
    print("max_eps = "+str(max_eps))
    print("min_samples = "+str(min_samples))
    db = DBSCAN(eps=max_eps,min_samples=max_minSamples).fit(X)
    labels = db.labels_
    while i < len(labels):
        if labels[i] != -1:
            labels[i] = 0
        else:
            labels[i] = 1
        i = i + 1
    writeLabel(labels)
    print(str(scores_data))
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
	for opt,arg in opts:
		if opt in ("-i","--iforest"):
			iForest(X_training_scaled)
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