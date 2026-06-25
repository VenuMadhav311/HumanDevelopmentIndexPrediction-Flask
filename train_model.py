import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("dataset/HDI.csv", sep="\t")

X = df.drop("HDI", axis=1)
X = X.drop("Country", axis=1)

y = df["HDI"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = model.score(X_test, y_test)
print("Model Accuracy:", score)

with open("model/HDI.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")