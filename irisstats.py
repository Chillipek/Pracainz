import pandas as pd
from pandas import DataFrame
from sklearn.datasets import load_iris

iris = load_iris()

df=pd.DataFrame(iris.data, columns=iris.feature_names)

minvalue=df.min()
maxvalue=df.max()
medianvalue=df.median()
meanvalue=df.mean()
print(f'Min values:\n{minvalue}')
print(f'Min values:\n{maxvalue}')
print(f'Min values:\n{medianvalue}')
print(f'Min values:\n{meanvalue}')