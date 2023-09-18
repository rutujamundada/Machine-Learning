# In this application we create our own algorithm for classified machine learning
# We create our own K Nearest Neighbour algorithm

# Consider below characteristics of Machine Learning Appliction

# Classifier       :    User Defined K Nearest Neigbour
# Dataset          :    Iris Dataset
# Features         :    Sepal Length,Sepal Width,Petal Length,Petal Width
# Labels           :    Versicolor , Setosa , Virginica
# Training Dataset :    75 Entries
# Testing Dataset  :    75 Entries

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class MarvelousKNN():

    def fit(self,TrainingData,TrainingTarget):
        self.TrainingData=TrainingData
        self.TrainingTarget=TrainingTarget

    def predict(self,TestData):
        predictions=[]
        for row in TestData:
            lebel=self.closest(row)
            predictions.append(lebel)
        return predictions

    def closest(self,row):
        bestdistance=euc(row,self.TrainingData[0])
        bestindex=0
        for i in range(1,len(self.TrainingData)):
            dist=euc(row,self.TrainingData[i])
            if dist < bestdistance:
                bestdistance=dist
                bestindex=i
        return self.TrainingTarget[bestindex]


def MarvelousKNNeighbour():
    border = "-"*50

    iris=load_iris()

    data=iris.data
    target=iris.target

    for i in range(len(iris.target)):
        print("ID:%d,Label:%s ,Feature:%s"%(i,iris.data[i],iris.target[i]))
    print("Size of actual data set:%d"%(i+1))

    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)

    print(border)
    print("--------------Training Data-------------------")
    for i in range(len(data_train)):
        print("ID :%d,Label : %s , Feature : %s" % (i, data_train[i], target_train[i]))
    print("Size of actual data set:%d" % (i + 1))

    print(border)
    print("--------------Testing Data-------------------")
    for i in range(len(data_test)):
        print("ID :%d,Label : %s , Feature : %s" % (i, data_test[i], target_test[i]))
    print("Size of actual data set:%d" % (i + 1))
    print(border)

    classifier=MarvelousKNN()

    classifier.fit(data_train,target_train)

    predictions=classifier.predict(data_test)

    Accuracy=accuracy_score(target_test,predictions)

    return Accuracy

def main():
    Accuracy=MarvelousKNNeighbour()
    print("Accuracy is: ",Accuracy*100,"%")

if __name__ == "__main__":
    main()