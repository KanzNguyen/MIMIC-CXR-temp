import os
import argparse
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from clip_dataset import create_clip_dataloaders
from clip_model import ClipDualView
from clip_eval import compute_embeddings, retrieval_metrics, linear_probe

try:
    import wandb
except Exception:
    wandb = None


def info_nce_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits = logit_scale * image_emb @ text_emb.t()
    labels = torch.arange(image_emb.size(0), device=image_emb.device)
    loss_i2t = nn.CrossEntropyLoss()(logits, labels)
    loss_t2i = nn.CrossEntropyLoss()(logits.t(), labels)
    return (loss_i2t + loss_t2i) * 0.5


def group_parameters(model: ClipDualView, base_lr_img: float, base_lr_text: float, base_lr_head: float, weight_decay: float):
    visual_params = []
    text_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('model.visual'):
            visual_params.append(param)
        elif name.startswith('model.transformer') or name.startswith('model.token_embedding') or name.startswith('model.positional_embedding') or name.startswith('model.text_projection'):
            text_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        { 'params': visual_params, 'lr': base_lr_img, 'weight_decay': weight_decay },
        { 'params': text_params, 'lr': base_lr_text, 'weight_decay': weight_decay },
        { 'params': head_params, 'lr': base_lr_head, 'weight_decay': weight_decay },
    ]
    return param_groups


def save_checkpoint(state: Dict, save_dir: str, filename: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True, help='Base path containing images and CSV')
    parser.add_argument('--csv_path', type=str, default=None, help='CSV file path; if None uses base_path/mimic_cxr_aug_train.csv')
    parser.add_argument('--model_name', type=str, default='ViT-B-16', choices=['ViT-B-16', 'RN50'])
    parser.add_argument('--pretrained', type=str, default='openai')
    parser.add_argument('--fuse_type', type=str, default='avg', choices=['avg', 'concat'])
    parser.add_argument('--text_ctx_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--lr_img', type=float, default=2e-5)
    parser.add_argument('--lr_text', type=float, default=2e-5)
    parser.add_argument('--lr_head', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--save_dir', type=str, default='checkpoints_clip')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='runs/clip_training')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--eval_every', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    csv_path = args.csv_path or os.path.join(args.base_path, 'mimic_cxr_aug_train.csv')

    train_loader, val_loader, test_loader, num_classes = create_clip_dataloaders(
        csv_path=csv_path,
        base_path=args.base_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        text_ctx_len=args.text_ctx_len,
    )

    model = ClipDualView(model_name=args.model_name, pretrained=args.pretrained, fuse_type=args.fuse_type, device=device)
    model.set_trainable(image_encoder=True, text_encoder=True, projection_heads=True)

    scaler = GradScaler(enabled=args.use_amp)

    param_groups = group_parameters(model, args.lr_img, args.lr_text, args.lr_head, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_r1 = 0.0

    if args.use_wandb and wandb is not None:
        wandb.init(project='mimic-cxr-clip', config=vars(args))

    writer = SummaryWriter(log_dir=args.log_dir)

    if args.resume and os.path.exists(args.resume):
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state.get('scaler', scaler.state_dict()))
        start_epoch = state.get('epoch', 0) + 1
        best_val_r1 = state.get('best_val_r1', 0.0)

    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            img_f = batch['image_frontal'].to(device)
            img_l = batch['image_lateral'].to(device)
            txt = batch['text_tokens'].to(device)

            with autocast(enabled=args.use_amp):
                image_emb, text_emb, logit_scale = model(img_f, img_l, txt)
                loss = info_nce_loss(image_emb, text_emb, logit_scale)
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            total_loss += loss.item() * args.accum_steps

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        if args.use_wandb and wandb is not None:
            wandb.log({'train/loss': avg_loss, 'epoch': epoch})
        writer.add_scalar('train/loss', avg_loss, epoch)

        if (epoch + 1) % args.eval_every == 0:
            with torch.no_grad():
                img_emb_val, txt_emb_val, labels_val = compute_embeddings(model, val_loader, device)
                ret = retrieval_metrics(img_emb_val, txt_emb_val, ks=[1, 5, 10])
                lp = linear_probe(img_emb_val.to(device), labels_val.to(device) if labels_val is not None else None)
                val_r1 = ret.get('R@1_img2txt', 0.0)

            # Logging
            for k, v in ret.items():
                writer.add_scalar(f'val/{k}', v, epoch)
                if args.use_wandb and wandb is not None:
                    wandb.log({f'val/{k}': v, 'epoch': epoch})
            for k, v in lp.items():
                writer.add_scalar(f'val/{k}', v, epoch)
                if args.use_wandb and wandb is not None:
                    wandb.log({f'val/{k}': v, 'epoch': epoch})

            # Save best
            if val_r1 > best_val_r1:
                best_val_r1 = val_r1
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'best_val_r1': best_val_r1,
                    'args': vars(args),
                }, args.save_dir, 'best.pt')

        # Save last
        save_checkpoint({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best_val_r1': best_val_r1,
            'args': vars(args),
        }, args.save_dir, 'last.pt')

    writer.close()


if __name__ == '__main__':
    main()
