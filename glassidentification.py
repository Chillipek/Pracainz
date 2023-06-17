import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# wczytanie danych “Glass Identification”
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
names = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'type']
df = pd.read_csv(url, names=names, index_col='id')

# podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df.drop('type', axis=1), df.type, test_size=0.2, random_state=42)

# utworzenie modelu drzewa decyzyjnego
model = DecisionTreeClassifier()

# dopasowanie modelu do danych treningowych
model.fit(X_train, y_train)

# przewidywanie na zbiorze testowym
y_pred = model.predict(X_test)

# ocena wyników
print(classification_report(y_test, y_pred))
