import pandas as pd

# wczytanie danych WBCD
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
names = ['id', 'diagnosis', 'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension', 'se_radius', 'se_texture', 'se_perimeter', 'se_area', 'se_smoothness', 'se_compactness', 'se_concavity', 'se_concave_points', 'se_symmetry', 'se_fractal_dimension', 'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness', 'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']
df = pd.read_csv(url, names=names)

# usunięcie kolumny z identyfikatorem obserwacji
df.drop('id', axis=1, inplace=True)

# zamiana wartości tekstowych na numeryczne
df['diagnosis'] = pd.factorize(df['diagnosis'])[0]


minvalue=df.min()
maxvalue=df.max()
medianvalue=df.median()
meanvalue=df.mean()

print(f'Min values:\n{minvalue}')
print(f'Max values:\n{maxvalue}')
print(f'Median values:\n{medianvalue}')
print(f'Mean values:\n{meanvalue}')