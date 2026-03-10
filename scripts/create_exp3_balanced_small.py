import pandas as pd

df = pd.read_csv("exp3_train_unified.csv")

# Split by source
kaggle_df = df[df["source"] == "kaggle"]
mendeley_df = df[df["source"] == "mendeley"]
multi_df = df[df["source"] == "multi version"]

# Sample 5000 Kaggle
kaggle_sample = kaggle_df.sample(n=5000, random_state=42)

# Oversample Mendeley to 5000
mendeley_sample = mendeley_df.sample(n=5000, replace=True, random_state=42)

# Oversample Multi to 5000
multi_sample = multi_df.sample(n=5000, replace=True, random_state=42)

balanced_df = pd.concat([kaggle_sample, mendeley_sample, multi_sample])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

balanced_df.to_csv("exp3_train_balanced_small.csv", index=False)

print("Balanced small dataset created.")
print(balanced_df["source"].value_counts())