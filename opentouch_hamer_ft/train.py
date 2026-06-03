#!/usr/bin/env python
import os
import sys

# Set OSMesa environment variables at the very beginning to prevent pyrender/EGL device ID issues
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['PYRENDER_PLATFORM'] = 'osmesa'

# Set absolute path for sys.argv[0] so that PyTorch Lightning DDP child spawns 
# will correctly launch this script regardless of working directory changes
sys.argv[0] = os.path.abspath(__file__)

import faulthandler
faulthandler.enable()

import argparse
import pytorch_lightning as pl
import cv2

# CRITICAL FIX for PyTorch DataLoader multiprocessing + OpenCV:
# Disable OpenCV's internal multithreading to prevent Segmentation Faults (-11) after fork()
cv2.setNumThreads(0)

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Add paths
ft_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(ft_dir, ".."))
sys.path.append(os.path.join(workspace_dir, "hamer"))
sys.path.append(ft_dir)

from hamer.models.hamer import HAMER
from dataset import OpenTouchHamerDataset

class OpenTouchHAMER(HAMER):
    def __init__(self, cfg, learning_rate=1e-5, freeze_backbone=True):
        # Initialize without loading rendering to save GPU memory and headless execution
        super().__init__(cfg, init_renderer=False)
        self.learning_rate = learning_rate
        
        # Override automatic optimization back to standard lightning
        self.automatic_optimization = True
        
        if freeze_backbone:
            print("Freezing the backbone network to prevent overfitting...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            print("Unfreezing ONLY the last 2 Transformer blocks of the backbone to prevent OOM...")
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze only the last 2 blocks for fine-tuning
            if hasattr(self.backbone, 'blocks'):
                for block in self.backbone.blocks[-2:]:
                    for param in block.parameters():
                        param.requires_grad = True
                
        # Keep the last layer/head trainable
        for param in self.mano_head.parameters():
            param.requires_grad = True
                
    def training_step(self, batch, batch_idx):
        output = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, output, train=True)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
        
    def tensorboard_logging(self, *args, **kwargs):
        # Override to be a no-op to prevent WebGL/OpenGL/Renderer crashes during training
        pass

    def configure_optimizers(self):
        # Apply layer-wise learning rates if backbone is unfrozen
        head_params = []
        backbone_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
                    
        # Backbone uses 10x smaller learning rate to preserve pretrained features
        optim_groups = [
            {'params': head_params, 'lr': self.learning_rate},
            {'params': backbone_params, 'lr': self.learning_rate * 0.1}
        ]
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            weight_decay=1e-4
        )
        
        # Calculate total stepping batches for Cosine Annealing
        total_steps = self.trainer.estimated_stepping_batches
        print(f"Total training steps for LR Scheduler: {total_steps}")
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=self.learning_rate * 0.01  # Decay to 1% of max LR
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Hamer on OpenTouch dataset (DDP Parallel)")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(workspace_dir, "hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"), help="Path to pretrained Hamer checkpoint")
    parser.add_argument("--split_json", type=str, default=os.path.join(workspace_dir, "evaluation/opentouch_splits.json"), help="Splits JSON path")
    parser.add_argument("--bbox_json", type=str, default=os.path.join(ft_dir, "opentouch_train_val_bboxes.json"), help="BBox JSON path")
    parser.add_argument("--data_dir", type=str, default=os.path.join(workspace_dir, "opentouch/data"), help="Data folder path")
    
    # Multi-GPU training support
    parser.add_argument("--gpus", type=str, default="4", help="GPU indices (comma-separated, e.g. 4,5)")
    
    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate (per GPU)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--no_freeze", action="store_true", help="Do not freeze Backbone layers (will use 10x smaller LR for backbone)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--quick_test", action="store_true", help="Run a quick test training on a tiny subset")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set CWD to hamer root for relative path safety in hamer config loaders
    hamer_root = os.path.join(workspace_dir, "hamer")
    os.chdir(hamer_root)
    
    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpus = len(args.gpus.split(","))
    
    # Apply linear learning rate scaling rule
    lr_scaled = args.lr
    if num_gpus > 1:
        lr_scaled = args.lr * num_gpus
        print(f"\n>>> Multi-GPU training detected. Scaling base learning rate linearly: {args.lr} * {num_gpus} = {lr_scaled}")
    
    # Load Hamer configuration and pretrained weights
    print(f"Loading pretrained Hamer weights from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    from pathlib import Path
    from hamer.configs import get_config
    
    model_cfg_path = str(Path(args.checkpoint).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg_path, update_cachedir=True)
    
    # Override config values for ViT backbone compatibility
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
        
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()
        
    print("Loading pretrained weights without renderer (init_renderer=False) to bypass pyrender...")
    pretrained_model = HAMER.load_from_checkpoint(
        args.checkpoint,
        strict=False,
        cfg=model_cfg,
        init_renderer=False
    )
    
    # Create OpenTouchHAMER model
    model = OpenTouchHAMER(
        cfg=model_cfg,
        learning_rate=lr_scaled,
        freeze_backbone=not args.no_freeze
    )
    
    # Transfer weights from pretrained model
    print("Transferring pretrained weights to OpenTouchHAMER model...")
    model.load_state_dict(pretrained_model.state_dict(), strict=False)
    
    # Create datasets
    train_dataset = OpenTouchHamerDataset(
        cfg=model_cfg,
        split="train",
        split_json=args.split_json,
        bbox_json=args.bbox_json,
        data_dir=args.data_dir,
        train=True
    )
    
    val_dataset = OpenTouchHamerDataset(
        cfg=model_cfg,
        split="val",
        split_json=args.split_json,
        bbox_json=args.bbox_json,
        data_dir=args.data_dir,
        train=False
    )
    
    # Quick test option: truncate datasets for testing execution speed and flow
    if args.quick_test:
        print("--- RUNNING QUICK TEST MODE ---")
        train_dataset.samples = train_dataset.samples[:64]
        val_dataset.samples = val_dataset.samples[:32]
        args.epochs = 1
        args.num_workers = 0
        print(f"Truncated datasets: Train={len(train_dataset)}, Val={len(val_dataset)}")
        
    # Define a worker_init_fn to specifically capture DataLoader worker Segmentation Faults
    def worker_init_fn(worker_id):
        import faulthandler
        import os
        # The log file will be saved in the hamer root directory since os.chdir(hamer_root) was called
        log_file = open(f"worker_crash_dump_pid{os.getpid()}.txt", "w")
        faulthandler.enable(file=log_file)

    # Create DataLoaders
    # Note: PyTorch Lightning DDP automatically applies DistributedSampler, 
    # dividing datasets among active GPU worker processes.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    # Set up checkpoints callback
    ckpt_dir = os.path.join(ft_dir, "checkpoints_test" if args.quick_test else "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best_ft_model",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Initialize Pytorch Lightning Trainer with DDP configuration for multi-GPU
    strategy = "ddp_find_unused_parameters_true" if num_gpus > 1 else "auto"
    print(f"Initializing PL Trainer | GPUs={num_gpus} ({args.gpus}) | Strategy={strategy} | Batch Size (per GPU)={args.batch_size}")
    
    # Setup Logger
    if args.use_wandb:
        print("🚀 Initializing Weights & Biases Logger...")
        logger = WandbLogger(project="opentouch-hamer-ft", name="hamer-finetune")
    else:
        logger = True
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=num_gpus,
        strategy=strategy,
        precision="16-mixed",  # Enable mixed precision for 2x speedup and 50% less VRAM
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    
    print("\nStarting Fine-tuning loop...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print(f"\nFine-tuning completed. Best checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
