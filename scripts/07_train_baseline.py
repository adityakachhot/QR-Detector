import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report
from tqdm import tqdm

# =========================================
# DEVICE (Apple MPS or CPU)
# =========================================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =========================================
# CUSTOM DATASET
# =========================================

class QRDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["filepath"]
        label = self.data.iloc[idx]["label"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = 1 if label == "malicious" else 0

        return image, torch.tensor(label, dtype=torch.float32)

# =========================================
# TRANSFORMS (ImageNet normalization)
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
# LOAD DATA (DEBUG MODE)
# =========================================

train_dataset = QRDataset("debug_train.csv", transform)
test_dataset = QRDataset("debug_test.csv", transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

# =========================================
# MODEL (EfficientNet-B0)
# =========================================

model = models.efficientnet_b0(pretrained=True)

# Replace final layer for binary classification
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

        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

# =========================================
# EVALUATION
# =========================================

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)

        outputs = model(images).squeeze()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, digits=4))
# =========================================
# SAVE MODEL
# =========================================

torch.save(model.state_dict(), "efficientnet_kaggle.pth")
print("Model saved as efficientnet_kaggle.pth")