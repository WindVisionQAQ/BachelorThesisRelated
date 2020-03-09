import csv
import numpy as np
from sklearn.ensemble import IsolationForest

with open("trainingData.csv") as csvfile:
	csv_reader = csv.reader(csvfile)
	header = next(csv_reader)
vol_orig = header[4:-1]
for d in vol_orig:
	d = float(d)

with open("newTrainingData.csv") as csvfile:
	csv_reader = csv.reader(csvfile)
	header = next(csv_reader)
vol_new = header[4:-1]
for d in vol_new:
	d = float(d)

header = []
dataList = []
with open("newTrainingData.csv") as csvfile:
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

X_training_new = np.array(dataList)

header = []
dataList = []
with open("TrainingData.csv") as csvfile:
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

X_training = np.array(dataList)

for i in range(len(vol_new)):
    if i<4:
	    X_training_new[:,i] = X_training[:,]*np.exp(vol_new(i)-vol_orig(i))
    else:
        X_training_new[:,i] = X_training[:,i]*((vol_new(i)-0.4)/(vol_orig(i)-0.4))^2


def iForest(X):
    rng = np.random.RandomState(6324)
    num_estimators = 50
    max_sample_num = 256
    clf = IsolationForest(n_estimators=num_estimators, n_jobs=1, behaviour='new', max_samples=max_sample_num,
                          random_state=rng, contamination='auto')
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
    writeLabelScore(label, scores)
    return

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
	with open("labelData_new_testV.csv","wb") as csvfile:
		csv_writer = csv.writer(csvfile)
		csv_writer.writerow(header)
		csv_writer.writerows(rows)
		csvfile.close()
	return

iForest(X_training_new)