#iris dataset using decision tree

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris=load_iris()

print("features of iris dataset")
print(iris.feature_names)

print("target names of iris dataset")
print(iris.target_names)

test_index=[1,51,101]

train_target=np.delete(iris.target,test_index)
train_data=np.delete(iris.data,test_index,axis=0)

test_target=iris.target[test_index]
test_data=iris.data[test_index]

classifier=tree.DecisionTreeClassifier()

classifier.fit(train_data,train_target)

print("values we removed for testing")
print(test_index)

print("output")
print(classifier.predict(test_data))