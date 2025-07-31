import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = "./data/iris.csv"
df = pd.read_csv(iris)
X, y = df.iloc[:,:-1],df.iloc[:,-1]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'models/iris_model.pkl')
