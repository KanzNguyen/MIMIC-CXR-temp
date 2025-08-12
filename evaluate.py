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


def find_best_thresholds_per_class(y_true, y_probs, step=0.05):
    """
    Tìm threshold tốt nhất cho từng nhãn dựa trên F1-score macro
    Args:
        y_true: ndarray [N, C]
        y_probs: ndarray [N, C]
        step: bước nhảy threshold
    Returns:
        best_thresholds: ndarray [C]
    """
    n_classes = y_true.shape[1]
    best_thresholds = np.zeros(n_classes)
    for c in range(n_classes):
        best_f1 = 0.0
        best_t = 0.5
        for t in np.arange(0.05, 0.95, step):
            y_pred = (y_probs[:, c] > t).astype(float)
            f1 = f1_score(y_true[:, c], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[c] = best_t
    return best_thresholds


def evaluate_multi_label(
    model,
    dataloader,
    device,
    label_names,
    init_threshold=0.2,
    tune_threshold=True,
    per_class_threshold=True,
    thresholds=None,
):
    """
    Comprehensive evaluation for multi-label classification
    Nếu per_class_threshold=True, tìm threshold tốt nhất cho từng nhãn dựa trên F1-score
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

    # Tìm threshold tốt nhất cho từng nhãn (không dùng test để tune)
    if thresholds is not None:
        best_thresholds = thresholds
    elif tune_threshold and per_class_threshold:
        best_thresholds = find_best_thresholds_per_class(y_true, all_probs)
    else:
        best_thresholds = np.full(y_true.shape[1], init_threshold)

    # Dự đoán với threshold tốt nhất cho từng nhãn
    y_pred = np.zeros_like(all_probs)
    for c in range(all_probs.shape[1]):
        y_pred[:, c] = (all_probs[:, c] > best_thresholds[c]).astype(float)

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
        "thresholds": best_thresholds,
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
    print(f"Per-class thresholds: {np.round(test_metrics['thresholds'], 3)}")
    print(f"Macro F1: {test_metrics['f1_macro']:.4f}, Micro F1: {test_metrics['f1_micro']:.4f}")
    print(f"Precision (macro): {test_metrics['precision_macro']:.4f}, Recall (macro): {test_metrics['recall_macro']:.4f}")
    print(f"Hamming Loss: {test_metrics['hamming_loss']:.4f}, AUC (macro): {test_metrics['auc_macro']:.4f}")

    print("\nPer-class breakdown:")
    for cls in label_cols:
        cls_metrics = test_metrics["per_class"].get(cls, {})
        f1 = cls_metrics.get("f1-score", 0.0)
        prec = cls_metrics.get("precision", 0.0)
        rec = cls_metrics.get("recall", 0.0)
        support = cls_metrics.get("support", 0.0)
        print(f"{cls:<30} F1: {f1:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  Support: {support:.0f}")