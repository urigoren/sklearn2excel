import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn2excel import translate_log_reg
df = pd.read_csv("../data/titanic.csv")
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
df = df[["Pclass", "Sex", "Age", "Survived", "Embarked"]].dropna()
model = LogisticRegression(solver="lbfgs")
model.fit(df[["Pclass", "Sex", "Age"]], df["Survived"])
print(translate_log_reg(model))
model.fit(df[["Pclass", "Sex", "Age", "Survived"]], df["Embarked"])
print(translate_log_reg(model))