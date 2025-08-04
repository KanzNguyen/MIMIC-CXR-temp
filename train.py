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


def train_model(model, train_loader, val_loader, optimizer, device, 
                num_epochs=50, patience=7, save_path='best_model.pth'):
    """
    Train model with early stopping
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        save_path: Path to save best model
        
    Returns:
        model: Trained model with best weights loaded
        history: Training history
    """
    best_f1 = 0
    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    print(f"Starting training for {num_epochs} epochs with patience {patience}")
    
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_f1 = evaluate(model, val_loader, device)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Check for improvement
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best F1: {best_f1:.4f} - Model saved!")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    # Load best weights
    model.load_state_dict(best_model_wts)
    print(f"Training completed. Best F1: {best_f1:.4f}")
    
    return model, history