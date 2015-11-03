import matplotlib.pyplot as plt
from sklearn import cross_validation
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
print("Image classified as " + str(clf.predict(X[177])) +"with label " +  str(y[177]))
plt.imshow(digits.images[177], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
