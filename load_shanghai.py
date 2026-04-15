import os
import shutil

sources = [
    'ShanghaiTech/part_A/train_data/images',
    'ShanghaiTech/part_B/train_data/images',
]

dest_train = 'my_data/train/images'
os.makedirs(dest_train, exist_ok=True)

total = 0
for source in sources:
    images = [f for f in os.listdir(source) if f.endswith('.jpg')]
    for img in images:
        src_path = os.path.join(source, img)
        part = 'SHA_A' if 'part_A' in source else 'SHA_B'
        new_name = f'{part}_{img}'
        dest_path = os.path.join(dest_train, new_name)
        shutil.copy2(src_path, dest_path)
        total += 1

print(f"Copied {total} ShanghaiTech images into {dest_train}")
