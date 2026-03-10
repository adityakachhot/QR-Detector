import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset_metadata.csv")

# ===============================
# EXPERIMENT 1 — Kaggle Baseline
# ===============================

kaggle_df = df[df["source"] == "kaggle"]

train_k, test_k = train_test_split(
    kaggle_df,
    test_size=0.2,
    stratify=kaggle_df["label"],
    random_state=42
)

train_k.to_csv("exp1_train_kaggle.csv", index=False)
test_k.to_csv("exp1_test_kaggle.csv", index=False)

print("Experiment 1 splits created.")

# ====================================
# EXPERIMENT 2 — Cross Dataset Testing
# ====================================

mendeley_df = df[df["source"] == "mendeley"]
multiversion_df = df[df["source"] == "multi version"]

kaggle_df.to_csv("exp2_train_kaggle.csv", index=False)
mendeley_df.to_csv("exp2_test_mendeley.csv", index=False)
multiversion_df.to_csv("exp2_test_multiversion.csv", index=False)

print("Experiment 2 splits created.")

# ====================================
# EXPERIMENT 3 — Unified Model
# ====================================

train_u, temp_u = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

val_u, test_u = train_test_split(
    temp_u,
    test_size=0.5,
    stratify=temp_u["label"],
    random_state=42
)

train_u.to_csv("exp3_train_unified.csv", index=False)
val_u.to_csv("exp3_val_unified.csv", index=False)
test_u.to_csv("exp3_test_unified.csv", index=False)

print("Experiment 3 splits created.")