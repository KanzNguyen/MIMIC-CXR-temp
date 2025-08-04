import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms


class LungXrayDataset(Dataset):
    def __init__(self, df, label_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_paths = df['image_path'].values
        self.labels = df[label_cols].values.astype('float32')
        self.transform = transform
        self.label_cols = label_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx])
        return image, label


def get_transforms():
    """Define transforms for training and validation"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform


def create_dataloaders(csv_path, base_path, batch_size=64, num_workers=2):
    """Create train, validation and test dataloaders"""
    # Read data
    df = pd.read_csv(csv_path)
    df['image_path'] = df.apply(lambda row: f"{base_path}/{row['split']}/{row['filename']}", axis=1)
    
    # Define label columns
    exclude_cols = ['filename', 'split', 'label', 'image_path']
    label_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Split data
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'valid']
    test_df = df[df['split'] == 'test']
    
    # Get transforms
    transform = get_transforms()
    
    # Create datasets
    train_dataset = LungXrayDataset(train_df, label_cols, transform)
    val_dataset = LungXrayDataset(val_df, label_cols, transform)
    test_dataset = LungXrayDataset(test_df, label_cols, transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, label_cols


def print_dataset_info(train_loader, val_loader, test_loader):
    """Print information about datasets"""
    print(f"Number of samples in the training dataset: {len(train_loader.dataset)}")
    print(f"Number of samples in the validation dataset: {len(val_loader.dataset)}")
    print(f"Number of samples in the test dataset: {len(test_loader.dataset)}")