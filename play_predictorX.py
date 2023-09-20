# Play predictor using k Nearest Neighbor

# The dataset used in this program contains information as weather we have decide whether to play or not

# Classifier       :    k Nearest Neigbor
# Dataset          :    Play predictor dataset
# Features         :    Weather , Temprature
# Labels           :    Yes,No
# Training Dataset :    30 Entries
# Testing Dataset  :    1 Entry

# import libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


def PlayPredictor(data_path):
    border = "-"*50

    # load data
    data=pd.read_csv(data_path)

    print("Size of actual dataset is",len(data))
    print(border)

    #clean,prepare and manipulate data
    feature_names=['Weather','Temperature']

    print("Name of features are :",feature_names)
    print(border)

    Weather=data.Weather
    Temprature=data.Temperature
    Play=data.Play

    #Creating Label Encoder
    le=preprocessing.LabelEncoder()

    #Converting string labels into numbers
    weather_encoded=le.fit_transform(Weather)
    print("Overcast : 0\nRainy : 1\nSunny : 2")
    print(border)
    print("Weather data after encoding :",weather_encoded)
    print(border)

    #Converting string labels into numbers
    temp_encoded=le.fit_transform(Temprature)
    label=le.fit_transform(Play)
    print("Cool : 0\nHot : 1\nMild : 2")
    print(border)
    print("Temperature data after encoding :",temp_encoded)
    print(border)

    #Combining weather and temp into single list tuples
    features=list(zip(weather_encoded,temp_encoded))
    print(border)
    print(features)
    print(border)

    #creating object ok knn
    model=KNeighborsClassifier(n_neighbors=3)

    #Train the model using training dataset
    model.fit(features,label)

    #testing the data
    predicted=model.predict([[0,2]])
    print("YES : 1\nNO : 0")
    print(border)
    print("RESULT",predicted)
    print(border)

def main():
    border = "-"*50
    print(border)
    print("Play predictor application using k Nearest Neighbor classifier")
    print(border)

    PlayPredictor("play_predictor.csv")

if __name__ == "__main__":
    main()