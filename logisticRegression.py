import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle as pk
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_csv('diabetes_012_health_indicators_updated.csv')
df = df.drop(['Unnamed: 0','AnyHealthcare', 'NoDocbcCost'], axis=1)

# Convert categorical variables to numerical
df['HighChol'].replace(['Yes', 'No'],[1, 0], inplace=True)
df['DiffWalk'].replace(['Yes', 'No'],[1, 0], inplace=True)
df['Veggies'].replace(['Yes', 'No'],[1, 0], inplace=True)
df['Fruits'].replace(['Yes', 'No'],[1, 0], inplace=True)
df['Smoker'].replace(['Yes', 'No'],[1, 0], inplace=True)
df['Sex'].replace(['male', 'female'],[1, 0], inplace=True)

# Define features and target
x = df.drop("Diabetes_012", axis=1)
y = df['Diabetes_012']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

def logisticregression():
    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Instantiate the Logistic Regression classifier
    logreg = LogisticRegression()

    # Train the classifier on the standardized training data
    logreg.fit(x_train_scaled, y_train)

    # Predict the target variable on the standardized test data
    y_pred = logreg.predict(x_test_scaled)

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100

    pk.dump(logreg, open('LogisticRegression.pkl', 'wb'))

    data = {'model': logreg, 'accuracy': accuracy}

    # Save the model to a pkl file
    with open('LogisticRegression.pkl', 'wb') as file:
        pk.dump(data, file)

  

logisticregression()