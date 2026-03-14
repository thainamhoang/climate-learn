from .components.cnn_blocks import PeriodicConv2D
from .components.pos_embed import get_2d_sincos_pos_embed
from .utils import register
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_
from typing import List, Optional, Dict

@register("vit")
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        history,
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        # NEW: Attention parameters
        save_attention: bool = False,
        attention_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.patch_size = patch_size
        # NEW: Attention saving config
        self.save_attention = save_attention
        self.attention_layers = attention_layers if attention_layers is not None else [-1]
        
        self.patch_embed = PatchEmbed(img_size, patch_size, self.in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    proj_drop=drop_rate,
                    attn_drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, out_channels * patch_size**2))
        self.head = nn.Sequential(*self.head)
        
        # NEW: Attention storage
        self.attention_weights: Dict[int, torch.Tensor] = {}
        self._hooks = []
        
        # NEW: Register hooks if enabled
        if self.save_attention:
            self._register_attention_hooks()
        
        self.initialize_weights()

    def _register_attention_hooks(self):
        """Register forward hooks to capture attention weights"""
        def _make_hook(layer_idx):
            def hook(module, input, output):
                # TIMM Block: output is (x, attn) when attn is stored
                if isinstance(output, tuple) and len(output) == 2:
                    attn_weights = output[1]  # [B, heads, N, N]
                    self.attention_weights[layer_idx] = attn_weights.detach().cpu()
            return hook
        
        for idx, block in enumerate(self.blocks):
            if idx in self.attention_layers or -1 in self.attention_layers:
                if hasattr(block, 'attn'):
                    hook = block.attn.register_forward_hook(_make_hook(idx))
                    self._hooks.append(hook)

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x: torch.Tensor):
        p = self.patch_size
        c = self.out_channels
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_encoder(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.flatten(1, 2)
        x = self.forward_encoder(x)
        x = self.head(x)
        preds = self.unpatchify(x)
        return preds

    # NEW: Attention retrieval methods
    def get_attention_weights(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        if layer_idx in self.attention_weights:
            return self.attention_weights[layer_idx]
        return None

    def get_all_attention_weights(self) -> Dict[int, torch.Tensor]:
        return self.attention_weights.copy()

    def clear_attention_weights(self):
        self.attention_weights.clear()

    def compute_attention_entropy(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        attn = self.get_attention_weights(layer_idx)
        if attn is not None:
            attn_mean = attn.mean(dim=2) + 1e-10
            entropy = -torch.sum(attn_mean * torch.log(attn_mean), dim=-1)
            return entropy
        return None