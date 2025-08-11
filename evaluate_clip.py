import os
import argparse
import torch
from clip_dataset import create_clip_dataloaders
from clip_model import ClipDualView
from clip_eval import compute_embeddings, retrieval_metrics, linear_probe, visualize_tsne


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_path', type=str, required=True)
    p.add_argument('--csv_path', type=str, default=None)
    p.add_argument('--model_name', type=str, default='ViT-B-16', choices=['ViT-B-16', 'RN50'])
    p.add_argument('--pretrained', type=str, default='openai')
    p.add_argument('--fuse_type', type=str, default='avg', choices=['avg', 'concat'])
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--text_ctx_len', type=int, default=256)
    p.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--save_vis', type=str, default='tsne.png')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    csv_path = args.csv_path or os.path.join(args.base_path, 'mimic_cxr_aug_train.csv')

    _, val_loader, test_loader, _ = create_clip_dataloaders(
        csv_path=csv_path,
        base_path=args.base_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        text_ctx_len=args.text_ctx_len,
    )

    loader = val_loader if args.split == 'val' else test_loader

    model = ClipDualView(model_name=args.model_name, pretrained=args.pretrained, fuse_type=args.fuse_type, device=device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()

    img_embs, txt_embs, labels = compute_embeddings(model, loader, device)

    ret = retrieval_metrics(img_embs, txt_embs, ks=[1, 5, 10])
    print('Retrieval metrics:')
    for k, v in ret.items():
        print(f'{k}: {v:.4f}')

    if labels is not None:
        lp = linear_probe(img_embs.to(device), labels.to(device))
        print('Linear probe:')
        for k, v in lp.items():
            print(f'{k}: {v:.4f}')

    if args.save_vis:
        visualize_tsne(img_embs, labels if labels is not None else None, args.save_vis)
        print(f'Saved t-SNE visualization to {args.save_vis}')


if __name__ == '__main__':
    main()
