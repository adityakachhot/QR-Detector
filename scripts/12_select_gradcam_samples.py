import pandas as pd

df = pd.read_csv("mendeley_model_comparison.csv")

samples = {}

# 1️⃣ Correct benign (both correct)
samples["both_correct_benign"] = df[
    (df["true_label"] == 0) &
    (df["kaggle_correct"] == True) &
    (df["balanced_correct"] == True)
].iloc[0]["filepath"]

# 2️⃣ Correct malicious (both correct)
samples["both_correct_malicious"] = df[
    (df["true_label"] == 1) &
    (df["kaggle_correct"] == True) &
    (df["balanced_correct"] == True)
].iloc[0]["filepath"]

# 3️⃣ Kaggle wrong, Balanced correct (benign case)
samples["kaggle_wrong_balanced_correct_benign"] = df[
    (df["true_label"] == 0) &
    (df["kaggle_correct"] == False) &
    (df["balanced_correct"] == True)
].iloc[0]["filepath"]

# 4️⃣ Kaggle wrong, Balanced correct (malicious case)
samples["kaggle_wrong_balanced_correct_malicious"] = df[
    (df["true_label"] == 1) &
    (df["kaggle_correct"] == False) &
    (df["balanced_correct"] == True)
].iloc[0]["filepath"]

selected_df = pd.DataFrame(samples.items(), columns=["type", "filepath"])
selected_df.to_csv("gradcam_selected_samples.csv", index=False)

print("Saved gradcam_selected_samples.csv")
print(selected_df)