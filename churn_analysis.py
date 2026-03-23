
# Loading data 
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")

# Churn count of customers 
print("Churn count:")
print(df["Exited"].value_counts()) # -> x times that we encounter the given value. 

# Average age of the customers

print("\nAverage age:")
print(df.groupby("Exited")["Age"].mean())
# groupby() -> compare clients those who churn and the rest. 
# mean() -> finds the average value 

# Custommers balance

print("\nAverage balance:")
print(df.groupby("Exited")["Balance"].mean())

# Porducts count 
print("\nProducts:")
print(df.groupby("Exited")["NumOfProducts"].mean())


print('---------------------------------------')

# Logic Regression
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
# model.fit() -> feeding the model with the data in order to learn. 

# Predictions
predictions = model.predict(X_test)
# model.predict() -> we want to see how to model whould predict data unseen until now,
# meaning we want the model to predict. y_test the right data do not feed the model with that. 


print("Predictions:")
print(predictions[:10])


print("Accuracy:", accuracy_score(y_test, predictions))
# % count how much of the predicted data is accured. 
print("Actual values:")
print(y_test[:10].values)
# If we want to compare the real data with the one that model predicted. 