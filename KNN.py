from sklearn import datasets
import numpy as np
import math
from collections import Counter

def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def KNN(data, labels, test,test_index, K = 5):
    neighbors = np.ones((2,K * 11))
    for x in range(0,neighbors.shape[1]):
        neighbors[0,x] = 0.001*x
        neighbors[1,x] = -1
    for x in range (0,data.shape[0]):
        score = cosine_similarity(data[x], test[test_index])
        if score > neighbors[0].min():
            index = np.where(neighbors[0] == neighbors[0].min())[0]
            index = index[0]
            neighbors[0,index] = score
            neighbors[1,index] = labels[x]
    labelsAsList = neighbors[1].tolist()
    counter = Counter(labelsAsList)
    return counter.most_common(1)[0][0]

testElemntsNo = 100
digits = datasets.load_digits()
X,y = digits.data[:-testElemntsNo], digits.target[:-testElemntsNo]
test, labels = digits.data[-testElemntsNo:], digits.target[-testElemntsNo:]
accuracy = 0 
for x in range(0,testElemntsNo):
    if KNN(X,y,test,x) - labels[x] == 0:
        accuracy += 1
print("accuracy = " + str((float(accuracy)/float(testElemntsNo)) * 100) + "%")