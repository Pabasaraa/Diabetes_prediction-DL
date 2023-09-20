import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle as pk

# Set the number of CPU cores
os.environ['LOKY_MAX_CPU_COUNT'] = '8'

# Load the dataset
df = pd.read_csv('diabetes_012_health_indicators_updated.csv')

# Drop unnecessary columns
df = df.drop(['Unnamed: 0', 'AnyHealthcare', 'NoDocbcCost'], axis=1)

# Replace categorical values with numerical ones
df['HighChol'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['DiffWalk'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['Veggies'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['Fruits'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['Smoker'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['Sex'].replace(['male', 'female'], [1, 0], inplace=True)

# Define features and target
x = df.drop("Diabetes_012", axis=1)
y = df['Diabetes_012']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

def kneighborsClassifier_bin():
    # Feature scaling
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(x_train)
    X_test = sc_x.transform(x_test)

    # Initialize K-NN
    classifier = KNeighborsClassifier(n_neighbors=450, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)

    # Predict the targets
    y_pred = classifier.predict(X_test)

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    pk.dump(classifier, open('KNeighborsClassifier.pkl', 'wb'))

    data = {'model': classifier, 'accuracy': accuracy}

    # Save the trained model to a pickle file
    with open('KNeighborsClassifier.pkl', 'wb') as file:
        pk.dump(data, file)


    
kneighborsClassifier_bin()
