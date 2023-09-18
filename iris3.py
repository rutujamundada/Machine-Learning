from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def MarvelousDecisionTree():

    #load the data
    iris=load_iris()

    #Initialize the data and target
    data=iris.data
    target=iris.target
            

    #split the data
    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5,random_state=42)

    #create the object of classifier
    classifier=tree.DecisionTreeClassifier()

    #train the data using fit
    classifier.fit(data_train,target_train)

    #test the output using predict
    predictions=classifier.predict(data_test)

    #calculate the accuracy
    Accuracy=accuracy_score(target_test,predictions)

    return Accuracy

#def MarvelouskNeighbourClassifier():


def main():

    acc=MarvelousDecisionTree()

    print("Accuracy using decision tree is",acc*100)

if __name__ == "__main__":
    main()