import json
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from PIL import Image

def generate_density_map(img_shape, points, sigma=15):
    density = np.zeros(img_shape[:2], dtype=np.float32)
    for x, y in points:
        if 0 <= int(y) < img_shape[0] and 0 <= int(x) < img_shape[1]:
            density[int(y), int(x)] += 1
    return gaussian_filter(density, sigma=sigma)

with open('via_annotations.json') as f:
    data = json.load(f)

img_data = data['_via_img_metadata']
output_dir = 'my_data/train/density_maps'
os.makedirs(output_dir, exist_ok=True)

success, failed = 0, 0
for key, val in img_data.items():
    filename = val['filename']
    regions  = val['regions']

    img_path = os.path.join('my_data/train/images', filename)
    if not os.path.exists(img_path):
        print(f'MISSING image: {filename}')
        failed += 1
        continue

    img = Image.open(img_path)
    img_shape = (img.height, img.width)

    points = []
    for r in regions:
        cx = r['shape_attributes']['cx']
        cy = r['shape_attributes']['cy']
        points.append((cx, cy))

    density = generate_density_map(img_shape, points)
    out_name = os.path.splitext(filename)[0] + '.npy'
    np.save(os.path.join(output_dir, out_name), density)

    success += 1
    print(f'✓ {filename} | heads: {len(points)} | saved: {out_name}')

print(f'\nDone. Success: {success} | Failed: {failed}')