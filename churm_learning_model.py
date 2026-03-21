import pandas as pd
from sklearn.metrics import accuracy_score
# 1. Зареждаме данните
df = pd.read_csv("Churn_Modelling.csv")

# 2. Подготвяме данните
X = df[["Age", "Balance", "NumOfProducts"]]
y = df["Exited"]

# 3. Разделяме данните
from sklearn.model_selection import train_test_split

# Разделяме данните на train и test, за да обучим модела 
# върху една част от данните и да проверим колко добре се справя върху нови, 
# невиждани данни.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Създаваме модела
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Предсказания
predictions = model.predict(X_test)

print("Predictions:")
print(predictions[:10])


print("Accuracy:", accuracy_score(y_test, predictions))
print("Actual values:")
print(y_test[:10].values)