from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
except ImportError:
    open_clip = None


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class ClipDualView(nn.Module):
    def __init__(
        self,
        model_name: str = 'ViT-B-16',
        pretrained: str = 'openai',
        fuse_type: str = 'avg',  # 'avg' or 'concat'
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if open_clip is None:
            raise ImportError("open_clip_torch is required. Please install open-clip-torch.")

        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.fuse_type = fuse_type
        self.embed_dim = self.model.text_projection.shape[1]

        if fuse_type == 'concat':
            self.fuse_proj = nn.Linear(self.embed_dim * 2, self.embed_dim)
        else:
            self.fuse_proj = None

        if device is not None:
            self.to(device)

    def encode_image_dual(self, img_frontal: torch.Tensor, img_lateral: torch.Tensor) -> torch.Tensor:
        img_f = self.model.encode_image(img_frontal)
        img_l = self.model.encode_image(img_lateral)
        # Both are normalized embeddings
        if self.fuse_type == 'avg':
            img = (img_f + img_l) * 0.5
            img = l2_normalize(img)
            return img
        else:  # concat
            fused = torch.cat([img_f, img_l], dim=-1)
            fused = self.fuse_proj(fused)
            fused = l2_normalize(fused)
            return fused

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        txt = self.model.encode_text(text_tokens)
        txt = l2_normalize(txt)
        return txt

    @property
    def logit_scale(self) -> torch.Tensor:
        # model.logit_scale is a parameter; use exp for scale per CLIP
        return self.model.logit_scale

    def forward(self, img_frontal: torch.Tensor, img_lateral: torch.Tensor, text_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_emb = self.encode_image_dual(img_frontal, img_lateral)
        text_emb = self.encode_text(text_tokens)
        logit_scale = self.model.logit_scale.exp()
        return image_emb, text_emb, logit_scale

    def set_trainable(self, image_encoder: bool = True, text_encoder: bool = True, projection_heads: bool = True):
        for name, param in self.model.named_parameters():
            if name.startswith('visual'):
                param.requires_grad = image_encoder
            elif name.startswith('transformer') or name.startswith('text'):
                param.requires_grad = text_encoder
            else:
                param.requires_grad = projection_heads
        if self.fuse_proj is not None:
            for p in self.fuse_proj.parameters():
                p.requires_grad = projection_heads
