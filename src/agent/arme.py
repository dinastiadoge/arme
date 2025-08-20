import pandas as pd
from sklearn.linear_model import LogisticRegression
# load dataset
df = pd.read_csv("\dataset.txt", sep="\n", header=None)
print(df)