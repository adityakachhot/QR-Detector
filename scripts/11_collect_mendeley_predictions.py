import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# =========================================
# DEVICE
# =========================================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =========================================
# DATASET
# =========================================

class QRDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[~self.data["filepath"].str.contains(".DS_Store", regex=False)]
        self.data = self.data.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["filepath"]
        label = 1 if row["label"] == "malicious" else 0

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32), img_path

# =========================================
# TRANSFORM
# =========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = QRDataset("exp2_test_mendeley.csv", transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# =========================================
# LOAD MODELS
# =========================================

def load_model(weight_path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model_kaggle = load_model("efficientnet_kaggle.pth")
model_balanced = load_model("efficientnet_exp3_balanced_small.pth")

# =========================================
# RUN INFERENCE
# =========================================

results = []

with torch.no_grad():
    for images, labels, paths in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Kaggle-only predictions
        outputs_k = model_kaggle(images).view(-1)
        probs_k = torch.sigmoid(outputs_k)
        preds_k = (probs_k > 0.5).int()

        # Balanced predictions
        outputs_b = model_balanced(images).view(-1)
        probs_b = torch.sigmoid(outputs_b)
        preds_b = (probs_b > 0.5).int()

        for i in range(len(paths)):
            results.append({
                "filepath": paths[i],
                "true_label": int(labels[i].item()),
                "kaggle_pred": int(preds_k[i].item()),
                "balanced_pred": int(preds_b[i].item())
            })

df_results = pd.DataFrame(results)

# Add correctness columns
df_results["kaggle_correct"] = df_results["true_label"] == df_results["kaggle_pred"]
df_results["balanced_correct"] = df_results["true_label"] == df_results["balanced_pred"]

df_results.to_csv("mendeley_model_comparison.csv", index=False)

print("Saved mendeley_model_comparison.csv")