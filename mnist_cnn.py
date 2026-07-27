"""
MNIST Handwritten Digit Recognition with CNN
Trains a 3-layer CNN on MNIST dataset using PyTorch and generates visualization plots.
Outputs: model_accuracy_loss.png, confusion_matrix.png, predictions.png
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device: use CPU (stable on macOS)
device = torch.device('cpu')

# ============================================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================================
print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: 10")

# ============================================================================
# 2. BUILD 3-LAYER CNN MODEL
# ============================================================================
print("\nBuilding 3-layer CNN model...")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Layer 1: Conv + ReLU + MaxPool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: Conv + ReLU + MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3: Conv + ReLU + Flatten + Dense
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================================================================
# 3. TRAIN MODEL FOR 5 EPOCHS
# ============================================================================
print("\nTraining model for 5 epochs...")
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(5):
    # Training phase
    model.train()
    train_loss, train_correct = 0.0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_loss /= len(train_dataset)
    train_acc = train_correct / len(train_dataset)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation phase
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(test_dataset)
    val_acc = val_correct / len(test_dataset)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/5 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ============================================================================
# 4. EVALUATE ON TEST SET
# ============================================================================
print("\nEvaluating on test set...")
model.eval()
test_loss, test_correct = 0.0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * labels.size(0)
        test_correct += (outputs.argmax(1) == labels).sum().item()

test_loss /= len(test_dataset)
test_accuracy = test_correct / len(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# 5. PLOT ACCURACY AND LOSS
# ============================================================================
print("\nGenerating accuracy/loss plot...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
axes[0].plot(train_accs, label='Train Accuracy', linewidth=2, marker='o')
axes[0].plot(val_accs, label='Validation Accuracy', linewidth=2, marker='s')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(train_losses, label='Train Loss', linewidth=2, marker='o')
axes[1].plot(val_losses, label='Validation Loss', linewidth=2, marker='s')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_accuracy_loss.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_accuracy_loss.png")
plt.close()

# ============================================================================
# 6. GENERATE CONFUSION MATRIX
# ============================================================================
print("\nGenerating confusion matrix...")
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
            xticklabels=range(10), yticklabels=range(10), cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - MNIST Test Set', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")
plt.close()

# ============================================================================
# 7. PREDICT ON 5 RANDOM TEST IMAGES
# ============================================================================
print("\nPredicting on 5 random test images...")
# Get random indices
random_indices = np.random.choice(len(test_dataset), 5, replace=False)

# Get images and labels
images_to_pred = []
true_labels_list = []
for idx in random_indices:
    img, label = test_dataset[idx]
    images_to_pred.append(img.unsqueeze(0))
    true_labels_list.append(label)

# Make predictions
model.eval()
predictions_list = []
with torch.no_grad():
    for img in images_to_pred:
        img = img.to(device)
        output = model(img)
        prob = torch.nn.functional.softmax(output, dim=1)[0]
        predictions_list.append(prob.cpu().numpy())

predicted_labels = [np.argmax(p) for p in predictions_list]

# Plot predictions
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
fig.suptitle('MNIST Predictions on Random Test Images', fontsize=14, fontweight='bold', y=1.02)

for idx, (ax, img_tensor) in enumerate(zip(axes, images_to_pred)):
    # Display image
    img_np = img_tensor.squeeze().numpy()
    ax.imshow(img_np, cmap='gray')
    ax.axis('off')

    # Color: green if correct, red if incorrect
    true_label = true_labels_list[idx]
    pred_label = predicted_labels[idx]
    confidence = predictions_list[idx][pred_label]

    is_correct = true_label == pred_label
    color = 'green' if is_correct else 'red'

    # Title with predicted and true labels
    title = f"Pred: {pred_label}\nTrue: {true_label}\nConf: {confidence:.2f}"
    ax.set_title(title, fontsize=10, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: predictions.png")
plt.close()

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Model Test Accuracy: {test_accuracy:.2%}")
print(f"Model Test Loss: {test_loss:.4f}")
print(f"\nPredictions on 5 random test images:")
for i in range(5):
    true_label = true_labels_list[i]
    pred_label = predicted_labels[i]
    confidence = predictions_list[i][pred_label]
    match = "✓" if true_label == pred_label else "✗"
    print(f"  {match} Image {i+1}: Predicted={pred_label}, True={true_label}, Confidence={confidence:.2%}")

print(f"\nOutput files:")
print(f"  • model_accuracy_loss.png")
print(f"  • confusion_matrix.png")
print(f"  • predictions.png")
print("="*60)
