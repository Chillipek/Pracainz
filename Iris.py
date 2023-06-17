import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# wczytanie danych iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# podział danych na zbiory, treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# tworzenie modelu klasyfikatora k-najbliższych sąsiadów
model = KNeighborsClassifier()

# dopasowanie modelu do danych treningowych
model.fit(X_train, y_train)

# przewidywanie na zbiorze testowym
y_pred = model.predict(X_test)

# ocena wyników
print(classification_report(y_test, y_pred))
