import pandas as pd 
from sklearn.linear_model import LogisticRegression

# load dataset
df = pd.read_csv("src/base/dataset.txt", sep="\t", header=None, names=["label", "message"])
pd.set_option("display.max_colwidth", None) #não é realmente necessário
print(df)