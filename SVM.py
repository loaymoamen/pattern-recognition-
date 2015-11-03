import matplotlib.pyplot as plt
from sklearn import cross_validation
from random import randint
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=10)
X,y = digits.data, digits.target
clf.fit(X,y)
#Applying cross validation with 10 folds of 180 picture each for 1797 images
scores = cross_validation.cross_val_score(clf, X, y, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#testing random image
r = randint(0,1796)
print("Image classified as " + str(clf.predict(X[r])) +" with label " +  str(y[r]))
plt.imshow(digits.images[r], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
