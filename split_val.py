import os
import shutil
import random

train_img = 'my_data/train/images'
train_den = 'my_data/train/density_maps'
val_img   = 'my_data/val/images'
val_den   = 'my_data/val/density_maps'

images = [f for f in os.listdir(train_img) if f.endswith(('.jpg', '.png'))]
random.shuffle(images)

val_count = int(len(images) * 0.2)
val_images = images[:val_count]

for img in val_images:
    # move image
    shutil.move(os.path.join(train_img, img), os.path.join(val_img, img))
    
    # move matching density map
    npy = os.path.splitext(img)[0] + '.npy'
    npy_src = os.path.join(train_den, npy)
    if os.path.exists(npy_src):
        shutil.move(npy_src, os.path.join(val_den, npy))

    print(f'Moved to val: {img}')

print(f'\nTrain images: {len(images) - val_count}')
print(f'Val images:   {val_count}')