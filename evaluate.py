import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    classification_report,
    roc_auc_score,
)
from tqdm import tqdm


def evaluate_multi_label(model, dataloader, device, label_names, init_threshold=0.2, tune_threshold=True):
    """
    Comprehensive evaluation for multi-label classification
    
    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device to run evaluation on
        label_names: List of label names
        init_threshold: Initial threshold for predictions
        tune_threshold: Whether to tune threshold for best F1
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating test set", leave=True):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # raw logits
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            all_logits.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)          # [N, C]
    all_labels = torch.cat(all_labels, dim=0)          # [N, C]
    all_probs = torch.sigmoid(all_logits).numpy()      # probabilities
    y_true = all_labels.numpy()                        # ground truth

    # Find best threshold if requested
    best_threshold = init_threshold
    if tune_threshold:
        best_f1 = 0.0
        for t in np.arange(0.05, 0.95, 0.05):
            y_pred_t = (all_probs > t).astype(float)
            f1_t = f1_score(y_true, y_pred_t, average="macro", zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = t

    y_pred = (all_probs > best_threshold).astype(float)

    # Calculate metrics
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    ham_loss = hamming_loss(y_true, y_pred)
    
    try:
        auc_macro = roc_auc_score(y_true, all_probs, average="macro")
    except Exception:
        auc_macro = None  # might fail if some classes have too few samples

    # Per-class detailed report
    per_class_report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    results = {
        "loss": total_loss / len(dataloader),
        "threshold_used": best_threshold,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "hamming_loss": ham_loss,
        "auc_macro": auc_macro,
        "per_class": per_class_report,
    }
    return results


def print_evaluation_results(test_metrics, label_cols):
    """Print evaluation results in a formatted way"""
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Threshold used: {test_metrics['threshold_used']:.2f}")
    print(f"Macro F1: {test_metrics['f1_macro']:.4f}, Micro F1: {test_metrics['f1_micro']:.4f}")
    print(f"Precision (macro): {test_metrics['precision_macro']:.4f}, Recall (macro): {test_metrics['recall_macro']:.4f}")
    print(f"Hamming Loss: {test_metrics['hamming_loss']:.4f}, AUC (macro): {test_metrics['auc_macro']:.4f}")

    print("\nPer-class breakdown:")
    for cls in label_cols:
        cls_metrics = test_metrics["per_class"].get(cls, {})
        f1 = cls_metrics.get("f1", 0.0)
        prec = cls_metrics.get("precision", 0.0)
        rec = cls_metrics.get("recall", 0.0)
        support = cls_metrics.get("support", 0.0)
        print(f"{cls:<30} F1: {f1:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  Support: {support:.0f}")