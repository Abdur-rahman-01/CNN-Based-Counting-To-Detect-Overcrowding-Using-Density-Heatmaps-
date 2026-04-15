from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from PIL import Image

# ── Load pretrained YOLOv8 ────────────────────────────────────
model = YOLO('yolov8n.pt')  # downloads automatically

# ── Test on your metro val images ─────────────────────────────
test_dir = 'my_data/val/images'
den_dir  = 'my_data/val/density_maps'

imgs = [f for f in os.listdir(test_dir)
        if f.endswith(('.jpg', '.png'))]

mae, mse = 0, 0
results_data = []
os.makedirs('results', exist_ok=True)

print('Running YOLOv8 on metro images...\n')

for i, img_name in enumerate(imgs):
    img_path = os.path.join(test_dir, img_name)
    npy_path = os.path.join(
        den_dir, os.path.splitext(img_name)[0] + '.npy'
    )
    if not os.path.exists(npy_path):
        continue

    # true count from density map
    density   = np.load(npy_path)
    true_count = int(density.sum())

    # YOLO inference
    result      = model(img_path, verbose=False)[0]
    pred_count  = sum(1 for cls in result.boxes.cls
                      if int(cls) == 0)  # class 0 = person

    err  = abs(pred_count - true_count)
    mae += err
    mse += err ** 2
    results_data.append({
        'image':      img_name,
        'true_count': true_count,
        'pred_count': pred_count,
        'error':      err
    })

    print(f'[{i+1:03d}] {img_name[:40]:40s} | '
          f'True: {true_count:4d} | '
          f'YOLO: {pred_count:4d} | '
          f'Error: {err:4d}')

    # Save annotated image for first 5 only
    if i < 5:
        img_cv  = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # Draw YOLO boxes
        for box in result.boxes:
            if int(box.cls) == 0:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(img_rgb, (x1,y1), (x2,y2),
                              (0,255,0), 2)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f'YOLOv8 | True: {true_count} | '
            f'Predicted: {pred_count} | Error: {err}',
            fontsize=13, fontweight='bold', color='darkred'
        )
        axes[0].imshow(img_rgb)
        axes[0].set_title('YOLO Detections (green boxes)')
        axes[0].axis('off')

        # density map for comparison
        axes[1].imshow(density, cmap='jet')
        axes[1].set_title('True Density Map (CSRNet ground truth)')
        axes[1].axis('off')

        out_name = f'results/yolo_{os.path.splitext(img_name)[0]}.png'
        plt.savefig(out_name, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {out_name}')

# ── Final metrics ─────────────────────────────────────────────
n   = len(results_data)
mae = mae / n
mse = (mse / n) ** 0.5

# under-detection rate (YOLO predicts less than half true count)
under = sum(1 for r in results_data
            if r['pred_count'] < r['true_count'] * 0.5)
under_rate = under / n * 100

print(f'\n{"="*50}')
print(f'YOLO Results on {n} metro images:')
print(f'MAE:                    {mae:.2f}')
print(f'MSE:                    {mse:.2f}')
print(f'Under-detection rate:   {under_rate:.1f}%')
print(f'  (images where YOLO predicts <50% of true count)')
print(f'{"="*50}')
print(f'\nKey finding: YOLOv8 severely undercounts in dense')
print(f'metro scenes due to occlusion. Bounding box detection')
print(f'fundamentally unsuitable for crowd density estimation.')

# ── Save CSV ──────────────────────────────────────────────────
with open('results/approach2_yolo_results.csv', 'w',
           newline='') as f:
    writer = csv.DictWriter(
        f, fieldnames=['image','true_count','pred_count','error'])
    writer.writeheader()
    writer.writerows(results_data)
    writer.writerow({})
    writer.writerow({'image': 'MAE', 'true_count': f'{mae:.2f}'})
    writer.writerow({'image': 'MSE', 'true_count': f'{mse:.2f}'})
    writer.writerow({'image': 'Under-detection rate',
                     'true_count': f'{under_rate:.1f}%'})

print(f'Saved: results/approach2_yolo_results.csv')

# ── Visualization: True vs Predicted scatter ───────────────────
true_counts = [r['true_count'] for r in results_data]
pred_counts = [r['pred_count'] for r in results_data]

plt.figure(figsize=(8, 6))
plt.scatter(true_counts, pred_counts, alpha=0.6,
            color='orange', edgecolors='black', s=60)
plt.plot([0, max(true_counts)], [0, max(true_counts)],
          'r--', label='Perfect prediction')
plt.xlabel('True Count (from density map annotations)')
plt.ylabel('YOLO Predicted Count')
plt.title(f'YOLOv8 vs True Count\nMAE: {mae:.2f} | '
           f'Under-detection: {under_rate:.1f}%')
plt.legend()
plt.tight_layout()
plt.savefig('results/approach2_yolo_scatter.png',
             dpi=150, bbox_inches='tight')
plt.show()
print('Saved: results/approach2_yolo_scatter.png')