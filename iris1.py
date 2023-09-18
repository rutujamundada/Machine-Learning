#Iris Case Study

from sklearn.datasets import load_iris

iris=load_iris()

print("Features names of iris datasets")
print(iris.feature_names)

print("Target names of iris datasets")
print(iris.target_names)

print(len(iris.target))
for i in range(len(iris.target)):
    print("ID:%d , Feature=%s , Label=%s "%(i,iris.data[i],iris.target[i]))
          