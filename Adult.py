import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

#wczytanie danych Adult
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv(url, names=names)

#usunięcie kolumny fnlwgt
df.drop('fnlwgt', axis=1, inplace=True)

#zamiana wartości tekstowych na numeryczne
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

#podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df.drop('income', axis=1), df.income, test_size=0.2, random_state=42)

#utworzenie modelu drzewa decyzyjnego
model = DecisionTreeClassifier(random_state=42)

#dopasowanie modelu do danych treningowych
model.fit(X_train, y_train)

#przewidywanie na zbiorze testowym
y_pred = model.predict(X_test)

#ocena wyników
print(classification_report(y_test, y_pred))
