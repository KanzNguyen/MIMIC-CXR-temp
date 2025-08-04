import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm
import copy


def train_one_epoch(model, dataloader, optimizer, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    loss_fn = nn.BCEWithLogitsLoss()

    pbar = tqdm(dataloader, desc="Training", leave=True)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, threshold=0.2):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    loss_fn = nn.BCEWithLogitsLoss()
    pbar = tqdm(dataloader, desc="Validating", leave=True)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)  
            preds = (probs > threshold).float()
            
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

            pbar.set_postfix(loss=loss.item())

    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()

    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(dataloader), f1


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, 
                num_epochs=100, patience=15, save_path='best_model.pth', grad_clip=1.0):
    """
    Train model for medical feature extraction (full fine-tuning)
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer (should be with lower LR for fine-tuning)
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Maximum number of epochs (more for full fine-tuning)
        patience: Early stopping patience (higher for medical data)
        save_path: Path to save best model
        grad_clip: Gradient clipping to prevent exploding gradients
        
    Returns:
        model: Trained model with best weights loaded
        history: Training history
    """
    best_f1 = 0
    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'lr': []}

    print(f"🚀 Starting medical feature extraction training")
    print(f"📊 Epochs: {num_epochs}, Patience: {patience}, Grad clip: {grad_clip}")
    
    for epoch in range(num_epochs):
        # Training với gradient clipping
        train_loss = train_one_epoch_with_clip(model, train_loader, optimizer, device, grad_clip)
        
        # Validation
        val_loss, val_f1 = evaluate(model, val_loader, device)
        
        # Update learning rate
        if scheduler:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"LR: {current_lr:.2e}")

        # Check for improvement
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best F1: {best_f1:.4f} - Model saved!")
        else:
            counter += 1
            if counter >= patience:
                print("🛑 Early stopping triggered!")
                break
        
        # Stop if learning rate becomes too low
        if current_lr < 1e-8:
            print("🛑 Learning rate too low - stopping training!")
            break

    # Load best weights
    model.load_state_dict(best_model_wts)
    print(f"🎯 Medical feature extraction training completed!")
    print(f"🏆 Best F1: {best_f1:.4f}")
    
    return model, history


def train_one_epoch_with_clip(model, dataloader, optimizer, device, grad_clip=1.0):
    """Train one epoch with gradient clipping"""
    model.train()
    total_loss = 0
    loss_fn = nn.BCEWithLogitsLoss()

    pbar = tqdm(dataloader, desc="Training", leave=True)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Gradient clipping để tránh exploding gradients khi fine-tune toàn bộ mô hình
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def create_medical_optimizer_scheduler(model, lr=1e-5):
    """
    Tạo optimizer và scheduler cho medical fine-tuning
    Learning rate thấp hơn để fine-tune cẩn thận
    """
    # Sử dụng AdamW với learning rate thấp cho fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,  # LR thấp cho fine-tuning toàn bộ mô hình
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # ReduceLROnPlateau để giảm LR khi không cải thiện
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=7,  # Patience thấp hơn cho fine-tuning
        min_lr=1e-8,
        verbose=True
    )
    
    return optimizer, scheduler