import argparse
import os
import torch
from torch import optim

from utils import set_seed, get_device
from dataset import create_dataloaders, print_dataset_info
from model import get_model
from train import train_model
from evaluate import evaluate_multi_label, print_evaluation_results


def main():
    parser = argparse.ArgumentParser(description='Lung X-ray Multi-label Classification')
    parser.add_argument('--csv_path', type=str, default='/kaggle/input/mimic-cxr/mimic-cxr.csv',
                        help='Path to CSV file')
    parser.add_argument('--base_path', type=str, default='/kaggle/input/mimic-cxr',
                        help='Base path to images')
    parser.add_argument('--model_name', type=str, default='resnet152',
                        choices=['resnet50', 'resnet101', 'resnet152'],
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_path', type=str, default='best_model.pth',
                        help='Path to save best model')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate, skip training')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, label_cols = create_dataloaders(
        csv_path=args.csv_path,
        base_path=args.base_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print_dataset_info(train_loader, val_loader, test_loader)
    print(f"Number of classes: {len(label_cols)}")
    print(f"Classes: {label_cols}")
    
    # Create model
    print(f"Creating {args.model_name} model...")
    model = get_model(num_classes=len(label_cols), model_name=args.model_name)
    model = model.to(device)
    
    if not args.eval_only:
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Train model
        print("Starting training...")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            patience=args.patience,
            save_path=args.save_path
        )
        
        print("Training completed!")
    else:
        # Load pre-trained model
        if os.path.exists(args.save_path):
            print(f"Loading model from {args.save_path}")
            model.load_state_dict(torch.load(args.save_path, map_location=device))
        else:
            print(f"Warning: Model file {args.save_path} not found. Using untrained model.")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_multi_label(
        model=model,
        dataloader=test_loader,
        device=device,
        label_names=label_cols,
        init_threshold=0.2,
        tune_threshold=True
    )
    
    # Print results
    print_evaluation_results(test_metrics, label_cols)


if __name__ == "__main__":
    main()