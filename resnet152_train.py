import torch
from utils import set_seed, get_device
from dataset import create_dataloaders, print_dataset_info
from model import get_model, print_model_info
from train import train_model, create_medical_optimizer_scheduler
from evaluate import evaluate_multi_label, print_evaluation_results
import argparse

# ========== CONFIG FOR MEDICAL FEATURE EXTRACTION ==========
BASE_PATH = '/kaggle/input/increased-generations'
MODEL_NAME = 'resnet152'
BATCH_SIZE = 64         
NUM_WORKERS = 2
NUM_EPOCHS = 5        
PATIENCE = 2          
LEARNING_RATE = 1e-5    
GRAD_CLIP = 1.0         # Gradient clipping để tránh exploding gradients
SEED = 42
SAVE_PATH = '/kaggle/working/best_medical_resnet152.pth'
EVAL_ONLY = False      
# ===========================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default=BASE_PATH, help='Path to image base directory')
    args = parser.parse_args()
    base_path = args.base_path
    csv_path = base_path.rstrip('/\\') + '/mimic-cxr.csv'
    print("\U0001F3E5 MEDICAL FEATURE EXTRACTION WITH RESNET152")
    print("\U0001F3AF Goal: Fine-tune entire ResNet152 to learn medical-specific features")
    
    # Set random seed
    set_seed(SEED)
    
    # Get device
    device = get_device()
    
    # Create data loaders
    print("\U0001F4CA Creating data loaders...")
    train_loader, val_loader, test_loader, label_cols, pos_weight = create_dataloaders(
        csv_path=csv_path,
        base_path=base_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    print_dataset_info(train_loader, val_loader, test_loader)
    print(f"Number of classes: {len(label_cols)}")
    print(f"Classes: {label_cols}")
    
    # Create model for medical feature extraction
    print(f"🏥 Creating {MODEL_NAME} for medical feature extraction...")
    model = get_model(num_classes=len(label_cols), model_name=MODEL_NAME)
    model = model.to(device)
    
    # Print model information
    print_model_info(model)
    
    if not EVAL_ONLY:
        # Create optimizer và scheduler cho medical fine-tuning
        optimizer, scheduler = create_medical_optimizer_scheduler(model, lr=LEARNING_RATE)
        
        # Train model để học medical-specific features
        print("🚀 Starting medical feature extraction training...")
        print("💡 This will fine-tune the ENTIRE ResNet152 for medical images")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=max(NUM_EPOCHS, 12),
            patience=max(PATIENCE, 5),
            save_path=SAVE_PATH,
            grad_clip=GRAD_CLIP,
            pos_weight=pos_weight,
            early_stop_metric='loss'
        )
        
        print("✅ Medical feature extraction training completed!")
    else:
        # Load pre-trained model
        import os
        if os.path.exists(SAVE_PATH):
            print(f"📂 Loading model from {SAVE_PATH}")
            model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        else:
            print(f"⚠️ Warning: Model file {SAVE_PATH} not found. Using untrained model.")
    
    # Evaluate: first find thresholds on validation, then apply to test (no leakage)
    print("🔬 Evaluating medical feature extraction on test set...")
    from evaluate import find_best_thresholds_per_class
    # Collect validation probabilities to tune thresholds
    model.eval()
    import torch.nn as nn
    import numpy as np
    all_logits, all_labels = [], []
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu())
            all_labels.append(labels.cpu())
    val_probs = torch.sigmoid(torch.cat(all_logits, dim=0)).numpy()
    val_true = torch.cat(all_labels, dim=0).numpy()
    best_thresholds = find_best_thresholds_per_class(val_true, val_probs)

    test_metrics = evaluate_multi_label(
        model=model,
        dataloader=test_loader,
        device=device,
        label_names=label_cols,
        init_threshold=0.2,
        tune_threshold=False,
        thresholds=best_thresholds
    )
    
    # Print results
    print_evaluation_results(test_metrics, label_cols)
    print(f"💾 Model saved at: {SAVE_PATH}")


if __name__ == "__main__":
    main()