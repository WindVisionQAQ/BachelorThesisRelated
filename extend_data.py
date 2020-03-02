import csv
import numpy as np
import math
from sklearn import preprocessing
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

def test100lines(c):
    res = np.zeros((100,1))
    for i in range(100):
        scale = 0
        print("processing " + str(i+200) + " lines")
        print(c[i + 200])
        while (c[i+200] < 1):
            c[i+200] *= 10
            scale = scale + 1
        res[i] = scale
        print(str(i+200) + " line finished.scale = " + str(scale))
    return res

def extractScale(column_data):
    res = np.zeros(column_data.shape)
    for i in range(column_data.shape[0]):
        scale=0
        print("processing "+str(i)+" lines")
        while(math.fabs(column_data[i])<1):
            column_data[i] *= 10
            scale = scale + 1
        res[i]=scale
        print(str(i)+" line finished.scale = "+str(scale))
    return res

def dataExtend(orig_data,pre_data):
    dim0 = orig_data.shape[0]
    print("dim0:"+str(dim0))
    dim1 = (orig_data.shape[1])*2 - 1
    print("dim1:"+str(dim1))
    extend_data = np.zeros((dim0,dim1))
    for i in range(dim1):
        if(i%2==0):
            extend_data[:,i] = pre_data[:,i/2]
        else:
            extend_data[:,i] = extractScale(extend_data[:,i-1])
    return extend_data

X_training = np.array(dataList)
X_training_scaled = preprocessing.scale(X_training)
X_extended = dataExtend(X_training, X_training_scaled)
assert (X_extended.shape[1] == 2 * X_training.shape[1] - 1)
header = ['0','0_scale','0.1','0.1_scale','0.2','0.2_scale','0.3','0.3_scale','0.5','0.5_scale','0.7','0.7_scale','0.8','0.8_scale','threshold']
with open('trainingData_extended.csv','wb') as csvfile:
	csv_writer = csv.writer(csvfile)
	csv_writer.writerow(header)
	csv_writer.writerows(X_extended)

