import torch
from utils import set_seed, get_device
from dataset import create_dataloaders, print_dataset_info
from model import get_model, print_model_info
from train import train_model, create_medical_optimizer_scheduler
from evaluate import evaluate_multi_label, print_evaluation_results

# ========== CONFIG FOR MEDICAL FEATURE EXTRACTION ==========
CSV_PATH = '/kaggle/input/mimic-cxr/mimic-cxr.csv'
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
    print("🏥 MEDICAL FEATURE EXTRACTION WITH RESNET152")
    print("🎯 Goal: Fine-tune entire ResNet152 to learn medical-specific features")
    
    # Set random seed
    set_seed(SEED)
    
    # Get device
    device = get_device()
    
    # Create data loaders
    print("📊 Creating data loaders...")
    train_loader, val_loader, test_loader, label_cols = create_dataloaders(
        csv_path=CSV_PATH,
        base_path=BASE_PATH,
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
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            save_path=SAVE_PATH,
            grad_clip=GRAD_CLIP
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
    
    # Evaluate on test set
    print("🔬 Evaluating medical feature extraction on test set...")
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
    print(f"💾 Model saved at: {SAVE_PATH}")


if __name__ == "__main__":
    main()