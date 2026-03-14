# module.py - MODIFIED FOR ATTENTION SAVING
from typing import Callable, List, Optional, Tuple, Union
from ..data.processing.era5_constants import CONSTANTS
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import pytorch_lightning as pl
import numpy as np
import os

class LitModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: LRScheduler,
        train_loss: Callable,
        val_loss: List[Callable],
        test_loss: List[Callable],
        train_target_transform: Optional[Callable] = None,
        val_target_transforms: Optional[List[Union[Callable, None]]] = None,
        test_target_transforms: Optional[List[Union[Callable, None]]] = None,
        # NEW: Attention parameters
        save_attention: bool = False,
        attention_save_dir: str = "attention_maps",
    ):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.test_loss = test_loss
        self.train_target_transform = train_target_transform
        # NEW: Attention saving config
        self.save_attention = save_attention
        self.attention_save_dir = attention_save_dir
        
        if val_target_transforms is not None:
            if len(val_loss) != len(val_target_transforms):
                raise RuntimeError(
                    "If 'val_target_transforms' is not None, its length must "
                    "match that of 'val_loss'. 'None' can be passed for "
                    "losses which do not require transformation. "
                )
            self.val_target_transforms = val_target_transforms
        if test_target_transforms is not None:
            if len(test_loss) != len(test_target_transforms):
                raise RuntimeError(
                    "If 'test_target_transforms' is not None, its length must "
                    "match that of 'test_loss'. 'None' can be passed for  "
                    "losses which do not rqeuire transformation. "
                )
            self.test_target_transforms = test_target_transforms
        
        self.mode = "direct"
        
        # NEW: Create attention save directory
        if self.save_attention:
            os.makedirs(self.attention_save_dir, exist_ok=True)

    def set_mode(self, mode):
        self.mode = mode

    def set_n_iters(self, iters):
        self.n_iters = iters

    def replace_constant(self, y, yhat, out_variables):
        for i in range(yhat.shape[1]):
            if out_variables[i] in CONSTANTS:
                yhat[:, i] = y[:, i]
        return yhat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y, in_variables, out_variables = batch
        yhat = self(x).to(device=y.device)
        yhat = self.replace_constant(y, yhat, out_variables)
        if self.train_target_transform:
            yhat = self.train_target_transform(yhat)
            y = self.train_target_transform(y)
        losses = self.train_loss(yhat, y)
        loss_name = getattr(self.train_loss, "name", "loss")
        loss_dict = {}
        if losses.dim() == 0:
            loss = losses
            loss_dict[f"train/{loss_name}:aggregate"] = loss
        else:
            for var_name, loss in zip(out_variables, losses):
                loss_dict[f"train/{loss_name}:{var_name}"] = loss
            loss = losses[-1]
            loss_dict[f"train/{loss_name}:aggregate"] = loss
        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        # NEW: Save attention on first validation batch
        if self.save_attention and batch_idx == 0:
            self._save_attention_maps(batch, "val", batch_idx)
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        # NEW: Save attention on first test batch
        if self.save_attention and batch_idx == 0:
            self._save_attention_maps(batch, "test", batch_idx)
        if self.mode == "direct":
            return self.evaluate(batch, "test")
        if self.mode == "iter":
            return self.evaluate_iter(batch, self.n_iters, "test")

    # NEW: Core attention saving method
    def _save_attention_maps(self, batch, stage: str, batch_idx: int):
        """Save attention weights to disk for visualization"""
        x, y, in_variables, out_variables = batch
        
        # Clear previous attention weights
        if hasattr(self.net, 'clear_attention_weights'):
            self.net.clear_attention_weights()
        
        # Forward pass to capture attention
        with torch.no_grad():
            _ = self(x.to(self.device))
        
        # Get and save attention weights
        if hasattr(self.net, 'get_all_attention_weights'):
            attention_weights = self.net.get_all_attention_weights()
            
            for layer_idx, attn in attention_weights.items():
                attn_np = attn.cpu().numpy()
                save_path = os.path.join(
                    self.attention_save_dir,
                    f"attention_{stage}_batch{batch_idx}_layer{layer_idx}.npy"
                )
                np.save(save_path, attn_np)
                
                # Log entropy metric
                if hasattr(self.net, 'compute_attention_entropy'):
                    entropy = self.net.compute_attention_entropy(layer_idx)
                    if entropy is not None:
                        self.log(f"{stage}/attention_entropy_layer{layer_idx}", 
                              entropy.mean().item(), on_epoch=True, sync_dist=True)
                
                print(f"💾 Saved: {save_path} (shape: {attn_np.shape})")

    def evaluate(self, batch, stage: str):
        x, y, in_variables, out_variables = batch
        yhat = self(x).to(device=y.device)
        yhat = self.replace_constant(y, yhat, out_variables)
        if stage == "val":
            loss_fns = self.val_loss
            transforms = self.val_target_transforms
        elif stage == "test":
            loss_fns = self.test_loss
            transforms = self.test_target_transforms
        else:
            raise RuntimeError("Invalid evaluation stage")
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            if transforms is not None and transforms[i] is not None:
                yhat_ = transforms[i](yhat)
                y_ = transforms[i](y)
            else:
                yhat_ = yhat
                y_ = y
            losses = lf(yhat_, y_)
            loss_name = getattr(lf, "name", f"loss_{i}")
            if losses.dim() == 0:
                loss_dict[f"{stage}/{loss_name}:agggregate"] = losses
            else:
                for var_name, loss in zip(out_variables, losses):
                    loss_dict[f"{stage}/{loss_name}:{var_name}"] = loss
                loss_dict[f"{stage}/{loss_name}:aggregate"] = losses[-1]
        self.log_dict(loss_dict, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(batch[0]))
        return loss_dict

    def evaluate_iter(self, batch, n_iters: int, stage: str):
        x, y, in_variables, out_variables = batch
        x_iter = x
        for _ in range(n_iters):
            yhat_iter = self(x_iter).to(device=x_iter.device)
            yhat_iter = self.replace_constant(y, yhat_iter, out_variables)
            x_iter = x_iter[:, 1:]
            x_iter = torch.cat((x_iter, yhat_iter.unsqueeze(1)), dim=1)
        yhat = yhat_iter
        if stage == "val":
            loss_fns = self.val_loss
            transforms = self.val_target_transforms
        elif stage == "test":
            loss_fns = self.test_loss
            transforms = self.test_target_transforms
        else:
            raise RuntimeError("Invalid evaluation stage")
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            if transforms is not None and transforms[i] is not None:
                yhat_t = transforms[i](yhat)
                y_t = transforms[i](y)
            else:
                yhat_t = yhat
                y_t = y
            losses = lf(yhat_t, y_t)
            loss_name = getattr(lf, "name", f"loss_{i}")
            if losses.dim() == 0:
                loss_dict[f"{stage}/{loss_name}:agggregate"] = losses
            else:
                for var_name, loss in zip(out_variables, losses):
                    loss_dict[f"{stage}/{loss_name}:{var_name}"] = loss
                loss_dict[f"{stage}/{loss_name}:aggregate"] = losses[-1]
        self.log_dict(loss_dict, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(batch[0]))
        return loss_dict

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y, *other_items = batch
        return self(x)

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                "scheduler": self.lr_scheduler,
                "monitor": self.trainer.favorite_metric,
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            }
        else:
            scheduler = self.lr_scheduler
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}