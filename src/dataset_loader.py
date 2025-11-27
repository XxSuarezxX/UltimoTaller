# src/dataset_loader.py
import os
import zipfile
import urllib.request
import random
import shutil

class DatasetLoader:
    def __init__(self, url, dataset_dir="dataset", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.url = url
        self.dataset_dir = dataset_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def download(self, zip_name="dataset.zip"):
        os.makedirs(self.dataset_dir, exist_ok=True)
        zip_path = os.path.join(self.dataset_dir, zip_name)
        if not os.path.exists(zip_path):
            print("Descargando dataset...")
            urllib.request.urlretrieve(self.url, zip_path)
            print(f"Descargado en: {zip_path}")
        else:
            print(f"Archivo ya existe: {zip_path}")
        return zip_path

    def extract(self, zip_path):
        print("Extrayendo dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_dir)
        print(f"Extraído en: {self.dataset_dir}")

    def split_dataset(self):
        print("Creando carpetas train/val/test...")
        all_classes = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]
        
        for split in ["train", "val", "test"]:
            for cls in all_classes:
                os.makedirs(os.path.join(self.dataset_dir, split, cls), exist_ok=True)

        for cls in all_classes:
            class_path = os.path.join(self.dataset_dir, cls)
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            random.shuffle(images)
            n = len(images)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)

            for i, img in enumerate(images):
                src = os.path.join(class_path, img)
                if i < n_train:
                    dst = os.path.join(self.dataset_dir, "train", cls, img)
                elif i < n_train + n_val:
                    dst = os.path.join(self.dataset_dir, "val", cls, img)
                else:
                    dst = os.path.join(self.dataset_dir, "test", cls, img)
                shutil.move(src, dst)

            # Opcional: eliminar carpeta original si está vacía
            if not os.listdir(class_path):
                os.rmdir(class_path)
        
        print("Split completado: train/val/test")
