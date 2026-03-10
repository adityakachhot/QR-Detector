import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report
from tqdm import tqdm
import os

# =========================================
# DEVICE
# =========================================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =========================================
# DATASET CLASS
# =========================================

class QRDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Remove hidden/system files from metadata
        self.data = self.data[~self.data["filepath"].str.contains(".DS_Store")]

        # Reset index after filtering
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["filepath"]
        label = self.data.iloc[idx]["label"]

        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError):
            # Skip corrupted image safely
            return self.__getitem__((idx + 1) % len(self.data))

        if self.transform:
            image = self.transform(image)

        label = 1 if label == "malicious" else 0

        return image, torch.tensor(label, dtype=torch.float32)

# =========================================
# TRANSFORMS (Must match training)
# =========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================
# LOAD MODEL (NO deprecated warning)
# =========================================

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

model.load_state_dict(torch.load("efficientnet_exp3_balanced_small.pth"))
model = model.to(device)
model.eval()

print("Model loaded successfully.\n")

# =========================================
# EVALUATION FUNCTION
# =========================================

def evaluate(csv_file):
    dataset = QRDataset(csv_file, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating {csv_file}"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).view(-1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\nResults for {csv_file}")
    print(classification_report(all_labels, all_preds, digits=4))
    print("-" * 60)

# =========================================
# RUN TESTS
# =========================================

evaluate("exp2_test_mendeley.csv")
evaluate("exp2_test_multiversion.csv")