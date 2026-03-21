
# Loading data 
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")

# Churn count of customers 
print("Churn count:")
print(df["Exited"].value_counts())

# Average age of the customers

print("\nAverage age:")
print(df.groupby("Exited")["Age"].mean())

# Custommers balance

print("\nAverage balance:")
print(df.groupby("Exited")["Balance"].mean())

# Porducts count 
print("\nProducts:")
print(df.groupby("Exited")["NumOfProducts"].mean())


print('---------------------------------------')

import pandas as pd
from sklearn.metrics import accuracy_score
# Loading the data 
df = pd.read_csv("Churn_Modelling.csv")

# Data preperiing
X = df[["Age", "Balance", "NumOfProducts"]]
y = df["Exited"]

# Spliting the data
from sklearn.model_selection import train_test_split

# Here we split the data train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Creating the model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

print("Predictions:")
print(predictions[:100])


print("Accuracy:", accuracy_score(y_test, predictions))
print("Actual values:")
print(y_test[:10].values)