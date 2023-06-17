import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# wczytanie danych Titanic
path = "/Users/filipczop/Pracainz/Titanic/titanic.csv"
df = pd.read_csv(path)

# usunięcie niepotrzebnych cech
df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# zamiana wartości tekstowych na numeryczne
df['Sex'] = pd.factorize(df['Sex'])[0]

# wypełnienie brakujących wartości
df.fillna(df.mean(), inplace=True)

# standaryzacja danych
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), df.Survived, test_size=0.2, random_state=42)

# utworzenie modelu sieci neuronowej
model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)

# dopasowanie modelu do danych treningowych
model.fit(X_train, y_train)

# przewidywanie na zbiorze testowym
y_pred = model.predict(X_test)

# ocena wyników
print(classification_report(y_test, y_pred))