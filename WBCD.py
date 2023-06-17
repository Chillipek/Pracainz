import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# wczytanie danych WBCD
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
names = ['id', 'diagnosis', 'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension', 'se_radius', 'se_texture', 'se_perimeter', 'se_area', 'se_smoothness', 'se_compactness', 'se_concavity', 'se_concave_points', 'se_symmetry', 'se_fractal_dimension', 'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness', 'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']
df = pd.read_csv(url, names=names)

# usunięcie kolumny z identyfikatorem obserwacji
df.drop('id', axis=1, inplace=True)

# zamiana wartości tekstowych na numeryczne
df['diagnosis'] = pd.factorize(df['diagnosis'])[0]

# standaryzacja danych
scaler = StandardScaler()
df.iloc[:,1:] = scaler.fit_transform(df.iloc[:,1:])

# podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df.drop('diagnosis', axis=1), df.diagnosis, test_size=0.2, random_state=42)

# utworzenie modelu sieci neuronowej
model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)

# dopasowanie modelu do danych treningowych
model.fit(X_train, y_train)

# przewidywanie na zbiorze testowym
y_pred = model.predict(X_test)

# ocena wyników
print(classification_report(y_test, y_pred))