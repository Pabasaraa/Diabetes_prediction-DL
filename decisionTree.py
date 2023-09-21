import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle as pk

# Load your dataset
df = pd.read_csv('diabetes_012_health_indicators_updated.csv')
df = pd.DataFrame(df)

# Drop unnecessary columns
df = df.drop(['Unnamed: 0', 'AnyHealthcare', 'NoDocbcCost'], axis=1)

# Replace categorical values with numerical values
df['HighChol'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['DiffWalk'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['Veggies'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['Fruits'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['Smoker'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['Sex'].replace(['male', 'female'], [1, 0], inplace=True)

# Define features and target
x = df.drop("Diabetes_012", axis=1)
y = df['Diabetes_012']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

def decisiontree():
    model = XGBClassifier()

    # Label encode categorical columns
    le = LabelEncoder()
    categ = ['HighChol', 'Fruits', 'Veggies', 'DiffWalk', 'Smoker', 'Sex']
    df[categ] = df[categ].apply(le.fit_transform)

    # Fit the XGBoost model on training data
    model.fit(x_train, y_train)

    # Predict the target variable on the standardized test data
    y_pred = model.predict(x_test)

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100

    pk.dump(model, open('XGBoost_model.pkl', 'wb'))

    data = {'model': model, 'accuracy': accuracy}

    # Save the model to a pkl file
    with open('XGBoost_model.pkl', 'wb') as file:
        pk.dump(data, file)

    # Predict the targets
    y_pred = model.predict(x_test)

decisiontree()
