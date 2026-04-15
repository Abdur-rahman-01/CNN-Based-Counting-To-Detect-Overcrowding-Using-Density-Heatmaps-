import os
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from PIL import Image

def mat_to_density(mat_path, img_shape, sigma=15):
    mat = sio.loadmat(mat_path)
    # correct key structure
    points = mat['image_info'][0][0][0][0][0]  # shape (N, 2)
    
    density = np.zeros(img_shape[:2], dtype=np.float32)
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= y < img_shape[0] and 0 <= x < img_shape[1]:
            density[y, x] += 1
    return gaussian_filter(density, sigma=sigma)

base = 'ShanghaiTech'
splits = [
    (f'{base}/part_A/train_data/images', f'{base}/part_A/train_data/ground-truth', f'{base}/part_A/train_data/density_maps'),
    (f'{base}/part_A/test_data/images',  f'{base}/part_A/test_data/ground-truth',  f'{base}/part_A/test_data/density_maps'),
    (f'{base}/part_B/train_data/images', f'{base}/part_B/train_data/ground-truth', f'{base}/part_B/train_data/density_maps'),
    (f'{base}/part_B/test_data/images',  f'{base}/part_B/test_data/ground-truth',  f'{base}/part_B/test_data/density_maps'),
]

for img_dir, gt_dir, out_dir in splits:
    os.makedirs(out_dir, exist_ok=True)
    imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    print(f'\nConverting {len(imgs)} images in {img_dir}')
    
    success, failed = 0, 0
    for img_file in imgs:
        num = img_file.replace('processed_IMG_', '').replace('.jpg', '')
        mat_path = os.path.join(gt_dir, f'GT_IMG_{num}.mat')
        
        if not os.path.exists(mat_path):
            print(f'  MISSING: GT_IMG_{num}.mat')
            failed += 1
            continue
        
        img = Image.open(os.path.join(img_dir, img_file))
        density = mat_to_density(mat_path, (img.height, img.width))
        out_path = os.path.join(out_dir, img_file.replace('.jpg', '.npy'))
        np.save(out_path, density)
        success += 1
        print(f'  ✓ IMG_{num} | heads: {int(density.sum())}')
    
    print(f'  Done: {success} converted, {failed} failed')

print('\nAll conversions complete.')