# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:14:09 2025

@author: Atef Kh
"""
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Daten-Transformationen (inkl. Augmentation)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # stronger augmentation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ]),
}

image_datasets = {
    'train': datasets.ImageFolder(
        r'.\train',
        data_transforms['train']
    ),
    'val': datasets.ImageFolder(
        r'.\valid',
        data_transforms['val']
    )
}
class_names = image_datasets['train'].classes

# 2. WeightedRandomSampler für Oversampling
targets = [label for _, label in image_datasets['train'].imgs]
class_sample_count = np.array([len(np.where(np.array(targets) == t)[0]) for t in np.unique(targets)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in targets])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=16, sampler=sampler),
    'val': DataLoader(image_datasets['val'], batch_size=16, shuffle=False)
}

# 3. Modell laden und anpassen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 Klassen: richtig/falsch
#checkpoint_path = r".\resnet18_best_model_alpha75_gamma3.pt"
#model = torch.load(checkpoint_path,weights_only=False)
model = model.to(device)

# 4. Verlustfunktion und Optimierer
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()
#criterion = FocalLoss()
class_counts = [len([img for img, label in image_datasets['train'].imgs if label == i]) for i in range(2)]
class_weights = torch.FloatTensor([sum(class_counts)/c for c in class_counts]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 5. Training 
num_epochs = 100
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
    epoch_loss = running_loss / len(image_datasets['train'])
    train_losses.append(epoch_loss)
    train_accuracies.append(correct_train / total_train)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_losses.append(val_loss / len(image_datasets['val']))
    val_accuracies.append(correct / total)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

# 6. Auswertung
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Klassifikationsbericht
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("resnet18_classification_report.csv", index=True)
print("Klassifikationsbericht gespeichert als resnet18_classification_report.csv")
print(report_df)

# Konfusionsmatrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("resnet18_confusion_matrix.png")
plt.show()
print("Konfusionsmatrix gespeichert als resnet18_confusion_matrix.png")

# Modell speichern
torch.save(model, r".\resnet50_best_model_alpha75_gamma3_v2.pt")
print("Modell gespeichert als resnet18_best_model.pth")

# Accuracy speichern
accuracy = accuracy_score(all_labels, all_preds)
with open("resnet18_accuracy.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
print(f"Accuracy: {accuracy:.4f} (gespeichert in resnet18_accuracy_2.txt)")

# Trainings- und Validierungskurven plotten
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig('resnet18_loss_curve.png')
plt.show()

plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.savefig('resnet18_accuracy_curve.png')
plt.show()