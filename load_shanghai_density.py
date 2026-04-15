import os
import shutil

sources = [
    ('ShanghaiTech/part_A/train_data/images',       'ShanghaiTech/part_A/train_data/density_maps',       'SHA_A'),
    ('ShanghaiTech/part_B/train_data/images',       'ShanghaiTech/part_B/train_data/density_maps',       'SHA_B'),
]

dest_img = 'my_data/train/images'
dest_den = 'my_data/train/density_maps'

os.makedirs(dest_img, exist_ok=True)
os.makedirs(dest_den, exist_ok=True)

total_img, total_den = 0, 0

for img_dir, den_dir, prefix in sources:
    imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    for img in imgs:
        new_name = f'{prefix}_{img}'
        
        # copy image (skip if already exists)
        dest_img_path = os.path.join(dest_img, new_name)
        if not os.path.exists(dest_img_path):
            shutil.copy2(os.path.join(img_dir, img), dest_img_path)
            total_img += 1

        # copy density map
        npy = img.replace('.jpg', '.npy')
        src_npy = os.path.join(den_dir, npy)
        dest_npy = os.path.join(dest_den, f'{prefix}_{npy}')
        if os.path.exists(src_npy) and not os.path.exists(dest_npy):
            shutil.copy2(src_npy, dest_npy)
            total_den += 1

print(f'Copied {total_img} images')
print(f'Copied {total_den} density maps')
print(f'Total density maps now: {len(os.listdir(dest_den))}')