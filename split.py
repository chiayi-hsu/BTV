
import os
import shutil
import random

def create_directory_structure(base_dir, classes):
    for dataset in ['train', 'val', 'test']:
        path = os.path.join(base_dir, dataset)
        os.makedirs(path, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(path, cls), exist_ok=True)

def split_dataset(base_dir, source_dir, classes, val_size=270, test_size=270):
    for cls in classes:
        class_path = os.path.join(source_dir, cls)
        images = os.listdir(class_path)
        random.shuffle(images)

        val_images = images[:val_size]
        test_images = images[val_size:val_size + test_size]
        train_images = images[val_size + test_size:]

        for img in train_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(base_dir, 'train', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
        for img in val_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(base_dir, 'val', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
        for img in test_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(base_dir, 'test', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)

source_dir = '/nas-hdd/prateek/data/EuroSAT_RGB'  # replace with the path to your dataset
base_dir = '/nas-hdd/prateek/data/EuroSAT_Splitted'  # replace with the path to the output directory

classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

create_directory_structure(base_dir, classes)
split_dataset(base_dir, source_dir, classes)