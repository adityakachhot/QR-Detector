import pandas as pd

# Load Kaggle experiment splits
train = pd.read_csv("exp1_train_kaggle.csv")
test = pd.read_csv("exp1_test_kaggle.csv")

# Create smaller subset for debugging
train_subset = train.sample(n=20000, random_state=42)
test_subset = test.sample(n=5000, random_state=42)

train_subset.to_csv("debug_train.csv", index=False)
test_subset.to_csv("debug_test.csv", index=False)

print("Debug subset created successfully.")
print("Train subset size:", len(train_subset))
print("Test subset size:", len(test_subset))