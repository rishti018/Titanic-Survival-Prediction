import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("Titanic-Dataset.csv")
print("First 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
features = ["Pclass", "Gender", "Age", "SibSp", "Parch", "Fare"]
data = df[features + ["Survived"]].copy()
data["Gender"] = data["Gender"].map({
    "male": 0,
    "female": 1
})
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Fare"] = data["Fare"].fillna(data["Fare"].median())
data = data.dropna()
X = data[features]
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=features,
    class_names=["Did Not Survive", "Survived"],
    filled=True,
    rounded=True
)
plt.title("Titanic Survival Prediction - Decision Tree")
plt.show()