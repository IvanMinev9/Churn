
# Loading data 
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")

# Колко клиента са напуснали
print("Churn count:")
print(df["Exited"].value_counts())

# Средна възраст
print("\nAverage age:")
print(df.groupby("Exited")["Age"].mean().round(2))

# Баланс
print("\nAverage balance:")
print(df.groupby("Exited")["Balance"].mean().round(2))

# Брой продукти
print("\nProducts:")
print(df.groupby("Exited")["NumOfProducts"].mean())
