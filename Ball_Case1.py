from sklearn import tree

def MarvelousML(weight,surface):
#rough 1
#smooth 0

    features=[[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]


#tennis 1
#cricket 2

    labels=[1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    obj=tree.DecisionTreeClassifier()

    obj=obj.fit(features,labels)

    result=obj.predict([[weight,surface]])

    if result==1:
        print("Your object looks like tennis ball")
    elif result==2:
        print("Your object looks like cricket ball")

def main():
    print("------Balls Case Study-----")

    print("Enter weight of object")
    weight=input()

    print("what is the surface type of your object rough or smooth")
    surface=input()

    if surface.lower()=="rough":
        surface=1
    elif surface.lower()=="smooth":
        surface=0
    else:
        print("Error:Wrong input")
        exit()

    MarvelousML(weight,surface)


if __name__ == "__main__":
    main()