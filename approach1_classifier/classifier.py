import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import csv

# ── Dataset ───────────────────────────────────────────────────
# CNN Classifier treats crowd counting as 3-class problem
# Low: 0-30 people, Medium: 31-80, High: 81+
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
                    # assign class label
                    if count < 30:
                        label = 0  # Low
                    elif count < 80:
                        label = 1  # Medium
                    else:
                        label = 2  # High
                    self.imgs.append(img)
                    self.labels.append(label)

        print(f'Dataset: {len(self.imgs)} images')
        print(f'Low: {self.labels.count(0)} | '
              f'Medium: {self.labels.count(1)} | '
              f'High: {self.labels.count(2)}')

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(
            os.path.join(self.img_dir, self.imgs[idx])
        ).convert('RGB')
        return self.transform(img), self.labels[idx]

# ── Model — ResNet18 classifier ───────────────────────────────
class CrowdClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.backbone.fc = nn.Linear(512, 3)  # 3 classes

    def forward(self, x):
        return self.backbone(x)

# ── Setup ─────────────────────────────────────────────────────
# Use your metro images only for fair comparison
train_set = CrowdClassifierDataset(
    '../my_data/train/images',
    '../my_data/train/density_maps'
)
test_set  = CrowdClassifierDataset(
    '../my_data/val/images',
    '../my_data/val/density_maps'
)

train_loader = DataLoader(train_set, batch_size=8,
                           shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1,
                           shuffle=False)

device    = torch.device('cpu')
model     = CrowdClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# ── Training ──────────────────────────────────────────────────
print('\nTraining CNN Classifier...\n')
for epoch in range(20):
    model.train()
    correct, total = 0, 0
    for imgs, labels in train_loader:
        imgs   = imgs.to(device)
        labels = torch.tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
    print(f'Epoch {epoch+1:02d}/20 | '
          f'Train Acc: {100.*correct/total:.1f}%')

# ── Evaluation ────────────────────────────────────────────────
print('\nEvaluating on test set...\n')
model.eval()
correct, total = 0, 0
all_preds, all_labels = [], []
mae = 0

# class midpoints for MAE estimation
midpoints = {0: 15, 1: 55, 2: 120}

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs   = imgs.to(device)
        labels = torch.tensor(labels).to(device)
        outputs  = model(imgs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
        all_preds.append(predicted.item())
        all_labels.append(labels.item())
        # MAE using class midpoints
        pred_count  = midpoints[predicted.item()]
        true_label  = labels.item()
        true_count  = midpoints[true_label]
        mae += abs(pred_count - true_count)

accuracy = 100. * correct / total
misclass = 100. - accuracy
mae      = mae / total

print(f'Test Accuracy:          {accuracy:.1f}%')
print(f'Misclassification Rate: {misclass:.1f}%')
print(f'Estimated MAE:          {mae:.1f}')
print(f'\nClass breakdown:')
print(f'  Low (0-30):    predicted {all_preds.count(0)} | '
      f'actual {all_labels.count(0)}')
print(f'  Medium (31-80): predicted {all_preds.count(1)} | '
      f'actual {all_labels.count(1)}')
print(f'  High (81+):    predicted {all_preds.count(2)} | '
      f'actual {all_labels.count(2)}')
print(f'\nKey finding: Classifier assigns coarse labels only.')
print(f'Cannot generate spatial density maps.')
print(f'Unsuitable for zone-level overcrowding detection.')

# Save results
os.makedirs('../results', exist_ok=True)
with open('../results/approach1_results.csv', 'w',
           newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    writer.writerow(['accuracy', f'{accuracy:.1f}'])
    writer.writerow(['misclassification_rate', f'{misclass:.1f}'])
    writer.writerow(['estimated_mae', f'{mae:.1f}'])

print('\nSaved: results/approach1_results.csv')