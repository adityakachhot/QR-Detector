import pandas as pd

df = pd.read_csv("exp3_train_unified.csv")
print(df["source"].value_counts())