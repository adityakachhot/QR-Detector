import pandas as pd

files = [
    "dataset_metadata.csv",
    "exp1_train_kaggle.csv",
    "exp1_test_kaggle.csv",
    "exp2_train_kaggle.csv",
    "exp2_test_mendeley.csv",
    "exp2_test_multiversion.csv",
    "exp3_train_unified.csv",
    "exp3_val_unified.csv",
    "exp3_test_unified.csv"
]

for file in files:
    df = pd.read_csv(file)

    # Remove hidden files
    df = df[~df["filepath"].str.contains(".DS_Store", regex=False)]

    df.to_csv(file, index=False)
    print(f"Cleaned {file}")

print("All CSV files cleaned successfully.")