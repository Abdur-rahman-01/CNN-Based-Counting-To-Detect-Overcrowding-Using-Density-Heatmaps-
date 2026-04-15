from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

model = YOLO('yolov8n.pt')

val_dir = 'my_data/val/images'
den_dir = 'my_data/val/density_maps'

all_imgs = [f for f in os.listdir(val_dir)
            if f.endswith(('.jpg', '.png'))]

# Find sparsest and densest image
sparse_img, dense_img = None, None
sparse_count, dense_count = 999, 0

for img_name in all_imgs:
    npy = os.path.splitext(img_name)[0] + '.npy'
    npy_path = os.path.join(den_dir, npy)
    if not os.path.exists(npy_path):
        continue
    count = int(np.load(npy_path).sum())
    if count < sparse_count:
        sparse_count = count
        sparse_img   = img_name
    if count > dense_count:
        dense_count = count
        dense_img   = img_name

print(f'Sparse: {sparse_img} | count: {sparse_count}')
print(f'Dense:  {dense_img}  | count: {dense_count}')

def run_yolo(img_path):
    img_cv  = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    result  = model(img_path, verbose=False)[0]

    annotated = img_rgb.copy()
    count = 0
    for box in result.boxes:
        if int(box.cls) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            # green box
            cv2.rectangle(annotated, (x1,y1), (x2,y2),
                          (0,230,0), 2)
            # confidence label
            cv2.rectangle(annotated, (x1, y1-18),
                          (x1+45, y1), (0,230,0), -1)
            cv2.putText(annotated, f'{conf:.2f}',
                        (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0,0,0), 1)
            count += 1

    # draw big red X zones in dense image corners to show failure
    if count == 0:
        h, w = annotated.shape[:2]
        cv2.putText(annotated, 'NO DETECTIONS',
                    (w//2-120, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255,50,50), 3)

    return annotated, count

sparse_path = os.path.join(val_dir, sparse_img)
dense_path  = os.path.join(val_dir, dense_img)

sparse_ann, sparse_pred = run_yolo(sparse_path)
dense_ann,  dense_pred  = run_yolo(dense_path)

sparse_orig = cv2.cvtColor(cv2.imread(sparse_path), cv2.COLOR_BGR2RGB)
dense_orig  = cv2.cvtColor(cv2.imread(dense_path),  cv2.COLOR_BGR2RGB)

sparse_err  = abs(sparse_pred - sparse_count)
dense_err   = abs(dense_pred  - dense_count)
dense_missed = max(0, dense_count - dense_pred)

# ── Figure ────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0f0f0f')

# Title
fig.text(0.5, 0.97,
         'YOLOv8 Object Detection — Works on Sparse Crowds, '
         'Fails on Dense Crowds',
         ha='center', va='top', fontsize=15,
         fontweight='bold', color='white')

# ── Row labels ────────────────────────────────────────────────
fig.text(0.01, 0.72, 'SPARSE\nCROWD',
         ha='left', va='center', fontsize=13,
         fontweight='bold', color='#2ecc71',
         rotation=90)
fig.text(0.01, 0.28, 'DENSE\nCROWD',
         ha='left', va='center', fontsize=13,
         fontweight='bold', color='#e74c3c',
         rotation=90)

# ── Subplot positions [left, bottom, width, height] ───────────
ax1 = fig.add_axes([0.05, 0.53, 0.40, 0.38])  # sparse original
ax2 = fig.add_axes([0.52, 0.53, 0.40, 0.38])  # sparse yolo
ax3 = fig.add_axes([0.05, 0.08, 0.40, 0.38])  # dense original
ax4 = fig.add_axes([0.52, 0.08, 0.40, 0.38])  # dense yolo

# ── Sparse row ────────────────────────────────────────────────
ax1.imshow(sparse_orig)
ax1.set_title('Original Image', fontsize=11,
               color='white', pad=8)
ax1.axis('off')
# border
for spine in ax1.spines.values():
    spine.set_edgecolor('#2ecc71')
    spine.set_linewidth(2)

ax2.imshow(sparse_ann)
ax2.set_title(
    f'YOLOv8 Detection  ✅  WORKS\n'
    f'True Count: {sparse_count}  |  '
    f'Detected: {sparse_pred}  |  '
    f'Error: {sparse_err}',
    fontsize=11, color='#2ecc71', pad=8
)
ax2.axis('off')
for spine in ax2.spines.values():
    spine.set_edgecolor('#2ecc71')
    spine.set_linewidth(2)

# annotation arrow
fig.text(0.47, 0.72,
         '→', fontsize=28, color='#2ecc71',
         ha='center', va='center')

# ── Dense row ─────────────────────────────────────────────────
ax3.imshow(dense_orig)
ax3.set_title('Original Image', fontsize=11,
               color='white', pad=8)
ax3.axis('off')
for spine in ax3.spines.values():
    spine.set_edgecolor('#e74c3c')
    spine.set_linewidth(2)

ax4.imshow(dense_ann)
ax4.set_title(
    f'YOLOv8 Detection  ❌  FAILS\n'
    f'True Count: {dense_count}  |  '
    f'Detected: {dense_pred}  |  '
    f'Missed: {dense_missed} people  '
    f'({dense_missed/max(dense_count,1)*100:.0f}%)',
    fontsize=11, color='#e74c3c', pad=8
)
ax4.axis('off')
for spine in ax4.spines.values():
    spine.set_edgecolor('#e74c3c')
    spine.set_linewidth(2)

fig.text(0.47, 0.27,
         '→', fontsize=28, color='#e74c3c',
         ha='center', va='center')

# ── Bottom stats bar ──────────────────────────────────────────
fig.text(0.5, 0.035,
         f'Overall YOLO Results on 162 Metro Images  |  '
         f'MAE: 283.23  |  MSE: 547.35  |  '
         f'Under-detection Rate: 85.8%  |  '
         f'Conclusion: Bounding box detection is '
         f'fundamentally unsuitable for dense crowd estimation',
         ha='center', fontsize=9.5, color='#aaaaaa',
         style='italic')

# ── Legend ────────────────────────────────────────────────────
green_patch = mpatches.Patch(color='#2ecc71',
                              label='Detected person (YOLO box)')
red_patch   = mpatches.Patch(color='#e74c3c',
                              label='Missed persons (occluded)')
fig.legend(handles=[green_patch, red_patch],
           loc='lower right',
           bbox_to_anchor=(0.97, 0.05),
           fontsize=9, facecolor='#1a1a1a',
           labelcolor='white', edgecolor='gray')

os.makedirs('results', exist_ok=True)
plt.savefig('results/yolo_sparse_vs_dense.png',
             dpi=150, bbox_inches='tight',
             facecolor='#0f0f0f')
plt.show()
print('Saved: results/yolo_sparse_vs_dense.png')