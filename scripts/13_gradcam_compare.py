import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models

# =========================================
# DEVICE
# =========================================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =========================================
# LOAD MODELS
# =========================================

def load_model(weight_path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model_kaggle = load_model("efficientnet_kaggle.pth")
model_balanced = load_model("efficientnet_exp3_balanced_small.pth")

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

# =========================================
# GRAD-CAM FUNCTION
# =========================================

def generate_gradcam(model, image_path, save_path):

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    image = Image.open(image_path).convert("RGB")
    original = np.array(image)
    input_tensor = transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)
    prob = torch.sigmoid(output)
    class_score = output.squeeze()

    model.zero_grad()
    class_score.backward()

    grads = gradients[0]
    acts = activations[0]

    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze()
    heatmap = heatmap.cpu().detach().numpy()

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(save_path, overlay)

    print(f"Saved Grad-CAM: {save_path}")

# =========================================
# RUN COMPARISON
# =========================================

os.makedirs("gradcam_outputs", exist_ok=True)

samples = pd.read_csv("gradcam_selected_samples.csv")

for _, row in samples.iterrows():
    img_path = row["filepath"]
    name = row["type"]

    generate_gradcam(
        model_kaggle,
        img_path,
        f"gradcam_outputs/{name}_kaggle.png"
    )

    generate_gradcam(
        model_balanced,
        img_path,
        f"gradcam_outputs/{name}_balanced.png"
    )