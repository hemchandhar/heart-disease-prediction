import pandas as pd
from glosys_ml.preprocessing import Heart_preprocessing
from glosys_ml.algorithms import Models

df=Heart_preprocessing.preprocessing('dataset/heart.csv')
x = df.iloc[:, :-1].values	
y= df.iloc[:, -1].values
y=y.reshape((y.shape[0],1))
a=int(input("\n\n\tModel selection:-\n\t\t\t1.Logistic Regression\n\t\t\t2.Naive bayes\n\t\t\t3.Decision Tree\n\t\t\t4.SVM classifier\n\t\t\t5.Random forest classifier\n\t\t\t6.Neural network(less accuracy)\n\nEnter your choice:"))

if(a==1):
	Models.logistic_regression(x,y)
elif(a==2):
	Models.naive_bayes(x,y)
elif(a==3):
	Models.decision_tree(x,y)
elif(a==4):
	Models.support_vector_machine(x,y)
elif(a==5):
	Models.random_forest(x,y)
elif(a==6):
	Models.neural_network(x,y)