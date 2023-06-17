from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=41)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

y_predict = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)

print("Accuracy: ", accuracy)