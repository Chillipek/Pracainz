import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv(url, names=names)

# Wybór kolumn zawierających dane liczbowe
numeric_columns = df.select_dtypes(include=['int64', 'float64'])

# Obliczenie wartości minimalnej, maksymalnej, mediany i średniej dla każdej kolumny
min_values = numeric_columns.min()
max_values = numeric_columns.max()
median_values = numeric_columns.median()
mean_values = numeric_columns.mean()

# Wyświetlenie wyników
print("Wartości minimalne:")
print(min_values)
print("\nWartości maksymalne:")
print(max_values)
print("\nMediana:")
print(median_values)
print("\nŚrednia wartość:")
print(mean_values)
