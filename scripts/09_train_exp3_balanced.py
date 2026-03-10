import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report
from tqdm import tqdm

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

        # Remove hidden files if any
        self.data = self.data[~self.data["filepath"].str.contains(".DS_Store", regex=False)]
        self.data = self.data.reset_index(drop=True)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["filepath"]
        label = row["label"]

        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError):
            return self.__getitem__((idx + 1) % len(self.data))

        if self.transform:
            image = self.transform(image)

        label = 1 if label == "malicious" else 0

        return image, torch.tensor(label, dtype=torch.float32)

# =========================================
# TRANSFORMS
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
# LOAD DATA
# =========================================

train_dataset = QRDataset("exp3_train_balanced_small.csv", transform)
val_dataset = QRDataset("exp3_val_unified.csv", transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))

# =========================================
# MODEL
# =========================================

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(device)

# =========================================
# LOSS & OPTIMIZER
# =========================================

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =========================================
# TRAINING LOOP
# =========================================

epochs = 3

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images).view(-1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

# =========================================
# VALIDATION
# =========================================

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validating"):
        images = images.to(device)

        outputs = model(images).view(-1)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nValidation Results:\n")
print(classification_report(all_labels, all_preds, digits=4))

# =========================================
# SAVE MODEL
# =========================================

torch.save(model.state_dict(), "efficientnet_exp3_balanced_small.pth")
print("Model saved as efficientnet_exp3_balanced_small.pth")