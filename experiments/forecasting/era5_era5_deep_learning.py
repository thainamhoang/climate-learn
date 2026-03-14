# era5_era5_deep_learning.py
from argparse import ArgumentParser
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import os
import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--checkpoint", default=None)

# NEW: Attention visualization arguments
parser.add_argument("--save_attention", action="store_true", 
                    help="Save attention maps during validation/test")
parser.add_argument("--attention_save_dir", type=str, default="attention_maps",
                    help="Directory to save attention maps")
parser.add_argument("--visualize_attention", action="store_true",
                    help="Generate visualization plots from saved attention maps")
parser.add_argument("--attention_layer", type=int, default=-1,
                    help="Which transformer layer to visualize (-1 for last)")

subparsers = parser.add_subparsers(
    help="Whether to perform direct, iterative, or continuous forecasting.",
    dest="forecast_type",
)

direct = subparsers.add_parser("direct")
iterative = subparsers.add_parser("iterative")
continuous = subparsers.add_parser("continuous")

direct.add_argument("era5_dir")
direct.add_argument("model", choices=["resnet", "unet", "vit"])
direct.add_argument("pred_range", type=int, choices=[6, 24, 72, 120, 240])

iterative.add_argument("era5_dir")
iterative.add_argument("model", choices=["resnet", "unet", "vit"])
iterative.add_argument("pred_range", type=int, choices=[6, 24, 72, 120, 240])

continuous.add_argument("era5_dir")
continuous.add_argument("model", choices=["resnet", "unet", "vit"])

args = parser.parse_args()

# Set up data
variables = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "relative_humidity",
    "specific_humidity",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "toa_incident_solar_radiation",
    "land_sea_mask",
    "orography",
    "lattitude",
]

in_vars = []
for var in variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            in_vars.append(var + "_" + str(level))
    else:
        in_vars.append(var)

if args.forecast_type in ("direct", "continuous"):
    out_variables = ["2m_temperature", "geopotential_500", "temperature_850"]
elif args.forecast_type == "iterative":
    out_variables = variables

out_vars = []
for var in out_variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            out_vars.append(var + "_" + str(level))
    else:
        out_vars.append(var)

if args.forecast_type in ("direct", "iterative"):
    dm = cl.data.IterDataModule(
        f"{args.forecast_type}-forecasting",
        args.era5_dir,
        args.era5_dir,
        in_vars,
        out_vars,
        src="era5",
        history=3,
        window=6,
        pred_range=args.pred_range,
        subsample=6,
        batch_size=128,
        num_workers=8,
    )
elif args.forecast_type == "continuous":
    dm = cl.data.IterDataModule(
        "continuous-forecasting",
        args.era5_dir,
        args.era5_dir,
        in_vars,
        out_vars,
        src="era5",
        history=3,
        window=6,
        pred_range=1,
        max_pred_range=120,
        random_lead_time=True,
        hrs_each_step=1,
        subsample=6,
        batch_size=128,
        buffer_size=2000,
        num_workers=8,
    )

dm.setup()

# Set up deep learning model
in_channels = 49
if args.forecast_type == "continuous":
    in_channels += 1  # time dimension
if args.forecast_type == "iterative":  # iterative predicts every var
    out_channels = in_channels
else:
    out_channels = 3

if args.model == "resnet":
    model_kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "n_blocks": 28,
    }
elif args.model == "unet":
    model_kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "ch_mults": (1, 2, 2),
        "is_attn": (False, False, False),
    }
elif args.model == "vit":
    model_kwargs = {
        "img_size": (32, 64),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "patch_size": 2,
        "embed_dim": 128,
        "depth": 8,
        "decoder_depth": 2,
        "learn_pos_emb": True,
        "num_heads": 4,
        # NEW: Attention saving parameters
        "save_attention": args.save_attention,
        "attention_layers": [args.attention_layer] if args.save_attention else None,
    }

optim_kwargs = {"lr": 5e-4, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
sched_kwargs = {
    "warmup_epochs": 5,
    "max_epochs": 50,
    "warmup_start_lr": 1e-8,
    "eta_min": 1e-8,
}

model = cl.load_forecasting_module(
    data_module=dm,
    model=args.model,
    model_kwargs=model_kwargs,
    optim="adamw",
    optim_kwargs=optim_kwargs,
    sched="linear-warmup-cosine-annealing",
    sched_kwargs=sched_kwargs,
    # NEW: Pass attention saving params to LitModule
    save_attention=args.save_attention,
    attention_save_dir=args.attention_save_dir,
)

# Setup trainer
pl.seed_everything(0)
default_root_dir = f"{args.model}_{args.forecast_type}_forecasting_{args.pred_range}"
logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")
early_stopping = "val/lat_mse:aggregate"
callbacks = [
    RichProgressBar(),
    RichModelSummary(max_depth=args.summary_depth),
    EarlyStopping(monitor=early_stopping, patience=args.patience),
    ModelCheckpoint(
        dirpath=f"{default_root_dir}/checkpoints",
        monitor=early_stopping,
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
    ),
]

trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    default_root_dir=default_root_dir,
    accelerator="gpu" if args.gpu != -1 else None,
    devices=[args.gpu] if args.gpu != -1 else None,
    max_epochs=args.max_epochs,
    strategy="ddp",
    precision="16",
)

# Define testing regime for iterative forecasting
def iterative_testing(model, trainer, args, from_checkpoint=False):
    for lead_time in [6, 24, 72, 120, 240]:
        n_iters = lead_time // args.pred_range
        model.set_mode("iter")
        model.set_n_iters(n_iters)
        test_dm = cl.data.IterDataModule(
            "iterative-forecasting",
            args.era5_dir,
            args.era5_dir,
            in_vars,
            out_vars,
            src="era5",
            history=3,
            window=6,
            pred_range=lead_time,
            subsample=1,
        )
        if from_checkpoint:
            trainer.test(model, datamodule=test_dm)
        else:
            trainer.test(model, datamodule=test_dm, ckpt_path="best")

# Define testing regime for continuous forecasting
def continuous_testing(model, trainer, args, from_checkpoint=False):
    for lead_time in [6, 24, 72, 120, 240]:
        test_dm = cl.data.IterDataModule(
            "continuous-forecasting",
            args.era5_dir,
            args.era5_dir,
            in_vars,
            out_vars,
            src="era5",
            history=3,
            window=6,
            pred_range=lead_time,
            max_pred_range=lead_time,
            random_lead_time=False,
            hrs_each_step=1,
            subsample=1,
            batch_size=128,
            buffer_size=2000,
            num_workers=8,
        )
        if from_checkpoint:
            trainer.test(model, datamodule=test_dm)
        else:
            trainer.test(model, datamodule=test_dm, ckpt_path="best")

# NEW: Attention visualization function
def visualize_attention_maps(attention_dir, img_size=(32, 64), patch_size=2):
    """Generate visualization plots from saved attention maps"""
    import glob
    
    attention_files = glob.glob(os.path.join(attention_dir, "*.npy"))
    
    if not attention_files:
        print(f"⚠️  No attention files found in {attention_dir}")
        return
    
    print(f"📊 Found {len(attention_files)} attention map files")
    
    # Create visualization directory
    viz_dir = os.path.join(attention_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    for attn_file in attention_files:
        # Load attention weights
        attn = np.load(attn_file)  # [B, heads, N, N]
        
        # Extract filename info
        filename = os.path.basename(attn_file).replace(".npy", "")
        
        # Average over batch and heads for visualization
        attn_mean = attn.mean(axis=(0, 1))  # [N, N]
        
        # Compute patch grid dimensions
        num_patches = attn_mean.shape[0]
        grid_h = img_size[0] // patch_size
        grid_w = img_size[1] // patch_size
        
        # Reshape to grid for spatial visualization
        attn_grid = attn_mean.reshape(grid_h, grid_w, num_patches)
        
        # Plot attention from first patch to all others
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Attention matrix
        plt.subplot(1, 3, 1)
        plt.imshow(attn_mean, cmap='viridis')
        plt.title(f"Attention Matrix\n{filename}")
        plt.xlabel("Key Patches")
        plt.ylabel("Query Patches")
        plt.colorbar(label="Attention Weight")
        
        # Subplot 2: First row (first patch attention to all)
        plt.subplot(1, 3, 2)
        first_row = attn_mean[0, :].reshape(grid_h, grid_w)
        plt.imshow(first_row, cmap='hot')
        plt.title("First Patch → All Patches")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.colorbar(label="Attention Weight")
        
        # Subplot 3: Attention entropy per patch
        plt.subplot(1, 3, 3)
        entropy = -np.sum(attn_mean * np.log(attn_mean + 1e-10), axis=1)
        entropy_grid = entropy.reshape(grid_h, grid_w)
        plt.imshow(entropy_grid, cmap='magma')
        plt.title("Attention Entropy\n(Lower = More Focused)")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.colorbar(label="Entropy")
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(viz_dir, f"{filename}.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved visualization: {viz_path}")
    
    print(f"\n🎨 All visualizations saved to: {viz_dir}")

# Train and evaluate model from scratch
if args.checkpoint is None:
    trainer.fit(model, datamodule=dm)
    if args.forecast_type == "direct":
        trainer.test(model, datamodule=dm, ckpt_path="best")
    elif args.forecast_type == "iterative":
        iterative_testing(model, trainer, args)
    elif args.forecast_type == "continuous":
        continuous_testing(model, trainer, args)
else:
    model = cl.LitModule.load_from_checkpoint(
        args.checkpoint,
        net=model.net,
        optimizer=model.optimizer,
        lr_scheduler=None,
        train_loss=None,
        val_loss=None,
        test_loss=model.test_loss,
        test_target_tranfsorms=model.test_target_transforms,
    )
    if args.forecast_type == "direct":
        trainer.test(model, datamodule=dm)
    elif args.forecast_type == "iterative":
        iterative_testing(model, trainer, args, from_checkpoint=True)
    elif args.forecast_type == "continuous":
        continuous_testing(model, trainer, args, from_checkpoint=True)

# NEW: Generate visualizations after training/testing
if args.visualize_attention:
    print("\n🎨 Generating attention visualizations...")
    visualize_attention_maps(
        args.attention_save_dir,
        img_size=model_kwargs.get("img_size", (32, 64)),
        patch_size=model_kwargs.get("patch_size", 2),
    )