import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ── CSRNet Model (same as training) ──────────────────────────
def make_layers(cfg, in_channels=3, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = make_layers([64, 64, 'M', 128, 128, 'M',
                                     256, 256, 256, 'M', 512, 512, 512])
        self.backend  = make_layers([512, 512, 512, 256, 128, 64],
                                     in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

# ── Load Model ────────────────────────────────────────────────
device = torch.device('cpu')
model  = CSRNet().to(device)
model.load_state_dict(torch.load('weights/csrnet_v3_best.pth',
                                  map_location='cpu'))
model.eval()
print('Model loaded successfully')

# ── Transform ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Inference Function ────────────────────────────────────────
def run_inference(image_path):
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        output = torch.relu(output)
        density_map = output.squeeze().numpy()

    raw_sum = density_map.sum()
    # correct scale based on actual image size
    img_w, img_h = img_pil.width, img_pil.height
    model_out_h, model_out_w = density_map.shape
    scale = (img_w * img_h) / (model_out_w * model_out_h)
    count = int(density_map.sum())
    print(f'Raw sum: {raw_sum:.4f} | Scale: {scale:.1f} | Count: {count}')

    if count < 20:
        status, color = 'NORMAL',       (0, 200, 0)
    elif count < 50:
        status, color = 'MODERATE',     (255, 165, 0)
    elif count < 100:
        status, color = 'CROWDED',      (255, 100, 0)
    else:
        status, color = 'OVERCROWDED',  (255, 0, 0)

    img_cv  = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    density_resized = cv2.resize(density_map,
                              (img_cv.shape[1], img_cv.shape[0]))
    p99 = np.percentile(density_resized, 99)
    p01 = np.percentile(density_resized, 1)
    density_clipped = np.clip(density_resized, p01, p99)
    density_norm = ((density_clipped - p01) / (p99 - p01 + 1e-8) * 255)
    density_norm = density_norm.astype(np.uint8)
    heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_rgb, 0.4, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Count: {count} | Status: {status}',
                  fontsize=16, fontweight='bold', color='darkred')
    axes[0].imshow(img_rgb);      axes[0].set_title('Original Image')
    axes[1].imshow(density_norm,
                   cmap='jet');   axes[1].set_title('Density Map')
    axes[2].imshow(overlay);      axes[2].set_title('Heatmap Overlay')
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = f'results/{img_name}_result.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Image:  {image_path}')
    print(f'Count:  {count} people')
    print(f'Status: {status}')
    print(f'Saved:  {out_path}')

# ── Run on test images ────────────────────────────────────────
test_images = [f for f in os.listdir('my_data/val/images')
               if f.endswith(('.jpg', '.png'))][:5]

for img in test_images:
    path = os.path.join('my_data/val/images', img)
    print(f'\nProcessing: {img}')
    run_inference(path)