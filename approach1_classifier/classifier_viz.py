import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ── Same Dataset class ────────────────────────────────────────
class CrowdClassifierDataset(Dataset):
    def __init__(self, img_dir, den_dir):
        self.img_dir = img_dir
        self.den_dir = den_dir
        self.imgs = []
        self.labels = []
        for img in os.listdir(img_dir):
            if img.endswith(('.jpg', '.png')):
                npy = os.path.splitext(img)[0] + '.npy'
                npy_path = os.path.join(den_dir, npy)
                if os.path.exists(npy_path):
                    density = np.load(npy_path)
                    count = density.sum()
                    if count < 30:   label = 0
                    elif count < 80: label = 1
                    else:            label = 2
                    self.imgs.append(img)
                    self.labels.append(label)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img = Image.open(
            os.path.join(self.img_dir, self.imgs[idx])
        ).convert('RGB')
        return self.transform(img), self.labels[idx]

# ── Same Model ────────────────────────────────────────────────
class CrowdClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(512, 3)
    def forward(self, x): return self.backbone(x)

# ── Retrain quickly ───────────────────────────────────────────
train_set = CrowdClassifierDataset(
    '../my_data/train/images',
    '../my_data/train/density_maps'
)
test_set = CrowdClassifierDataset(
    '../my_data/val/images',
    '../my_data/val/density_maps'
)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False)

device    = torch.device('cpu')
model     = CrowdClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

print('Training...')
for epoch in range(20):
    model.train()
    for imgs, labels in train_loader:
        imgs   = imgs.to(device)
        labels = torch.stack([l if isinstance(l, torch.Tensor) 
                              else torch.tensor(l) for l in labels])
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/20 done')

# ── Evaluation + Visuals ──────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
class_names = ['Low\n(0-30)', 'Medium\n(31-80)', 'High\n(81+)']

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        out  = model(imgs)
        _, pred = out.max(1)
        all_preds.append(pred.item())
        label = labels if isinstance(labels, int) else labels.item()
        all_labels.append(label)

accuracy  = sum(p==l for p,l in zip(all_preds,all_labels)) / len(all_labels) * 100
misclass  = 100 - accuracy

# ── Plot 1: Confusion Matrix ───────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names, ax=axes[0])
axes[0].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.1f}% | '
                   f'Misclassification: {misclass:.1f}%',
                   fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# ── Plot 2: Why classifier fails ──────────────────────────────
approaches = ['CNN\nClassifier', 'YOLOv8\n(pretrained)',
              'CSRNet\nPretrained', 'CSRNet\nv3']
mae_values = [55, 85, 50, 11.30]
colors     = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']

bars = axes[1].bar(approaches, mae_values, color=colors,
                    edgecolor='black', linewidth=0.8)
axes[1].set_title('MAE Comparison Across Approaches\n'
                   '(Lower is Better)', fontsize=12,
                   fontweight='bold')
axes[1].set_ylabel('Mean Absolute Error (MAE)')
axes[1].set_ylim(0, 100)

for bar, val in zip(bars, mae_values):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 1,
                  f'{val}', ha='center', fontweight='bold')

# Add "No density map" annotation
axes[1].annotate('No density\nmap output',
                  xy=(0, mae_values[0]),
                  xytext=(0.5, 70),
                  fontsize=9, color='red',
                  arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
os.makedirs('../results', exist_ok=True)
plt.savefig('../results/approach1_confusion_and_comparison.png',
             dpi=150, bbox_inches='tight')
plt.show()

print(f'\nAccuracy:          {accuracy:.1f}%')
print(f'Misclassification: {misclass:.1f}%')
print(f'Saved: results/approach1_confusion_and_comparison.png')

# Print classification report
print('\nDetailed Report:')
print(classification_report(all_labels, all_preds,
                              target_names=['Low','Medium','High']))