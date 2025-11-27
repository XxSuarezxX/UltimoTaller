from dataset_loader import DatasetLoader

# Ruta local al dataset
dataset_path = "./dataset"  # o la ruta absoluta

loader = DatasetLoader(dataset_path)
loader.split_dataset()
