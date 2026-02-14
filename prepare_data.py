import json
import os
import shutil
from sklearn.model_selection import train_test_split

JSON_PATH = 'instances.json'
IMAGE_SOURCE = 'mini_coco' 
OUTPUT_BASE = 'dataset'

CAT_MAP = {1: 0, 3: 1, 17: 2, 18: 3, 62: 4} 

for split in ['train', 'val']:
    os.makedirs(f'{OUTPUT_BASE}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUTPUT_BASE}/labels/{split}', exist_ok=True)

with open(JSON_PATH, 'r') as f:
    coco_data = json.load(f)

valid_img_ids = {ann['image_id'] for ann in coco_data['annotations'] if ann['category_id'] in CAT_MAP}
images_to_process = [img for img in coco_data['images'] if img['id'] in valid_img_ids]


train_imgs, val_imgs = train_test_split(images_to_process, test_size=0.2, random_state=42)

def process(img_list, split_name):
    for img in img_list:
        img_id = img['id']
        file_name = img['file_name']
        
        lines = []
        for ann in coco_data['annotations']:
            if ann['image_id'] == img_id and ann['category_id'] in CAT_MAP:
                dw = 1. / img['width']
                dh = 1. / img['height']
                x = (ann['bbox'][0] + ann['bbox'][2] / 2.0) * dw
                y = (ann['bbox'][1] + ann['bbox'][3] / 2.0) * dh
                w = ann['bbox'][2] * dw
                h = ann['bbox'][3] * dh
                lines.append(f"{CAT_MAP[ann['category_id']]} {x} {y} {w} {h}")
        
        if lines:
            
            txt_name = file_name.replace('.jpg', '.txt')
            with open(f'{OUTPUT_BASE}/labels/{split_name}/{txt_name}', 'w') as f:
                f.write('\n'.join(lines))
            
            shutil.copy(f'{IMAGE_SOURCE}/{file_name}', f'{OUTPUT_BASE}/images/{split_name}/{file_name}')

print("Dönüştürme başlıyor...")
process(train_imgs, 'train')
process(val_imgs, 'val')
print("Bitti! 'dataset' klasörün hazır.")