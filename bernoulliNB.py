import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn import metrics

# Load the dataset
df = pd.read_csv('diabetes_012_health_indicators_updated.csv')
df = pd.DataFrame(df)

# Preprocessing
df = df.drop(['Unnamed: 0','AnyHealthcare', 'NoDocbcCost'], axis=1)


df['HighChol'].replace(['Yes', 'No'],[1, 0], inplace=True)
df['DiffWalk'].replace(['Yes', 'No'],[1, 0], inplace=True)
df['Veggies'].replace(['Yes', 'No'],[1, 0], inplace=True)
df['Fruits'].replace(['Yes', 'No'],[1, 0], inplace=True)
df['Smoker'].replace(['Yes', 'No'],[1, 0], inplace=True)
df['Sex'].replace(['male', 'female'],[1, 0], inplace=True)


x=df.drop("Diabetes_012", axis= 1)
y= df['Diabetes_012']

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Training the model
def naivebayes():
  clf = BernoulliNB()

  # training the model on training set
  clf.fit(x_train, y_train)

  # making predictions on the testing set
  y_pred = clf.predict(x_test)

  # calculate accuracy
  accuracy = accuracy_score(y_test, y_pred) * 100
  
  pk.dump(clf, open('BernoulliNB.pkl', 'wb'))

  data = {'model': clf, 'accuracy': accuracy}
  
  with open('BernoulliNB.pkl', 'wb') as file:
     pk.dump(data, file)

naivebayes()
