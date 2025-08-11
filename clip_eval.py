from typing import Dict, List, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def compute_embeddings(model, dataloader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_img = []
    all_txt = []
    all_lbl = []
    with torch.no_grad():
        for batch in dataloader:
            img_f = batch['image_frontal'].to(device)
            img_l = batch['image_lateral'].to(device)
            txt = batch['text_tokens'].to(device)
            labels = batch.get('labels', None)
            image_emb, text_emb, _ = model(img_f, img_l, txt)
            all_img.append(image_emb.cpu())
            all_txt.append(text_emb.cpu())
            if labels is not None:
                all_lbl.append(labels)
    image_embs = torch.cat(all_img, dim=0)
    text_embs = torch.cat(all_txt, dim=0)
    labels = torch.cat(all_lbl, dim=0) if len(all_lbl) > 0 else None
    return image_embs, text_embs, labels


def retrieval_metrics(image_embs: torch.Tensor, text_embs: torch.Tensor, ks: List[int] = [1, 5, 10]) -> Dict[str, float]:
    sim = image_embs @ text_embs.t()  # cosine since both normalized
    ranks_img2txt = sim.argsort(dim=1, descending=True)
    gt = torch.arange(sim.size(0))

    results: Dict[str, float] = {}
    for k in ks:
        correct = (ranks_img2txt[:, :k] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
        results[f'R@{k}_img2txt'] = correct

    ranks_txt2img = sim.t().argsort(dim=1, descending=True)
    for k in ks:
        correct = (ranks_txt2img[:, :k] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
        results[f'R@{k}_txt2img'] = correct
    return results


def linear_probe(image_embs: torch.Tensor, labels: torch.Tensor, num_epochs: int = 3, lr: float = 1e-2, weight_decay: float = 0.0) -> Dict[str, float]:
    if labels is None:
        return {}
    num_samples, embed_dim = image_embs.shape
    num_classes = labels.shape[1]
    device = image_embs.device

    model = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    x = image_embs
    y = labels.to(device)
    for _ in range(num_epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().numpy()
    y_true = y.cpu().numpy()

    macro_f1 = f1_score(y_true, probs > 0.5, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, probs > 0.5, average='micro', zero_division=0)

    # mAP over classes
    ap_per_class = []
    for c in range(num_classes):
        ap = average_precision_score(y_true[:, c], probs[:, c])
        if not np.isnan(ap):
            ap_per_class.append(ap)
    mAP = float(np.mean(ap_per_class)) if len(ap_per_class) > 0 else 0.0

    return {
        'linear_probe_macro_f1': macro_f1,
        'linear_probe_micro_f1': micro_f1,
        'linear_probe_mAP': mAP,
    }


def visualize_tsne(image_embs: torch.Tensor, labels: torch.Tensor, save_path: str, perplexity: float = 30.0, n_iter: int = 1000):
    x = image_embs.cpu().numpy()
    y = labels.cpu().numpy() if labels is not None else None

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, learning_rate='auto', init='random')
    x2d = tsne.fit_transform(x)

    plt.figure(figsize=(8, 8))
    if y is not None:
        # color by max label index for visualization simplicity
        color = y.argmax(axis=1)
        scatter = plt.scatter(x2d[:, 0], x2d[:, 1], c=color, cmap='tab10', s=5, alpha=0.7)
        plt.legend(*scatter.legend_elements(num=10), title="label")
    else:
        plt.scatter(x2d[:, 0], x2d[:, 1], s=5, alpha=0.7)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.title('t-SNE of Image Embeddings')
    plt.savefig(save_path, dpi=200)
    plt.close()
