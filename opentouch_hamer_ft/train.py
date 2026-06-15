#!/usr/bin/env python
import os
import sys

import argparse

# ==========================================================================================
# 🛑 核心黑魔法：源码感知 + 全局空间硬核注入补丁（地表最强终结版，完美解决一切 NameError）
# ==========================================================================================
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--render_platform', type=str, default='egl', choices=['egl', 'osmesa'], help='Rendering platform (egl or osmesa)')
_args, _ = _parser.parse_known_args()

os.environ['PYOPENGL_PLATFORM'] = _args.render_platform
os.environ['PYRENDER_PLATFORM'] = _args.render_platform

try:
    import types
    import builtins
    import re
    import sys

    # 1. 定义一个全能通配符类：既是数字0，又是可任意调用的函数，还支持无限切片和属性延伸
    class UltimateMagicMock(int):
        def __call__(self, *args, **kwargs): return self
        def __getattr__(self, name): return self
        def __getitem__(self, item): return self
        def __iter__(self): return iter([])

    class PerfectMockModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith('__'): raise AttributeError(name)
            return UltimateMagicMock(0)

    mock_obj = PerfectMockModule('OpenGL.GL')

    # 2. 拦截系统的底层 __import__ 行为
    orig_import = builtins.__import__
    def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
        # 只要发现有任何文件在尝试染指 OpenGL/EGL/OSMesa
        if name.startswith('OpenGL') or name in ['EGL', 'OSMesa']:
            if globals is not None and '__file__' in globals:
                try:
                    # 【硬核注入】读取当前正在执行 import 的文件（如 texture.py）的源码
                    with open(globals['__file__'], 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # 抓取该文件里写的所有 OpenGL 相关的函数和常量（如 GL_TEXTURE_2D, glGenTextures 等）
                    tokens = re.findall(r'\b([gG][lL][A-Za-z0-9_]+|[eE][gG][lL][A-Za-z0-9_]+|OSMesa[A-Za-z0-9_]+)\b', content)
                    
                    # 直接强行把这些变量塞进该文件的全局命名空间，彻底断绝 NameError 的可能
                    for token in tokens:
                        if token not in globals:
                            globals[token] = UltimateMagicMock(0)
                except Exception:
                    pass
            return mock_obj
        return orig_import(name, globals, locals, fromlist, level)
    
    # 替换系统全局导入函数
    builtins.__import__ = custom_import

    # 3. 固化系统路由备份
    sys.modules['EGL'] = mock_obj
    sys.modules['OSMesa'] = mock_obj
    sys.modules['OpenGL'] = mock_obj
    sys.modules['OpenGL.GL'] = mock_obj
    sys.modules['OpenGL.GL.shaders'] = mock_obj
    
    print("\n====== [Success] Hardcore Global Token Injector Activated! ======\n")
except Exception as e:
    print(f"Bypass failed: {e}")
# ==========================================================================================

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
    def __init__(self, cfg, learning_rate=1e-5, freeze_backbone=True, teacher_model=None):
        # Initialize without loading rendering to save GPU memory and headless execution
        super().__init__(cfg, init_renderer=False)
        self.learning_rate = learning_rate
        
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        
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
            
    def forward_step(self, batch, train=False):
        # 先让当前正在训练的 Student 模型输出
        output = super().forward_step(batch, train=train)
        
        # 策略 3: 使用 Teacher 模型的相机预测直接覆盖 Student 的输出
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_output = self.teacher_model.forward_step(batch, train=False)
            output['pred_cam'] = teacher_output['pred_cam']
            if 'pred_cam_list' in teacher_output:
                output['pred_cam_list'] = teacher_output['pred_cam_list']
                
        return output
        
    def on_save_checkpoint(self, checkpoint):
        # 移除 teacher_model 权重以节省 Checkpoint 的存储空间
        if 'state_dict' in checkpoint:
            keys_to_remove = [k for k in checkpoint['state_dict'].keys() if k.startswith('teacher_model.')]
            for k in keys_to_remove:
                del checkpoint['state_dict'][k]
                
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
    parser.add_argument("--data_dir", type=str, default="/data/jiangrui/OpenTouch Data/extracted_dataset", help="Data folder path")
    
    # Multi-GPU training support
    parser.add_argument("--gpus", type=str, default="4", help="GPU indices (comma-separated, e.g. 4,5)")
    
    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate (per GPU)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--no_freeze", action="store_true", help="Do not freeze Backbone layers (will use 10x smaller LR for backbone)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name for saving checkpoints in a specific subfolder")
    parser.add_argument("--quick_test", action="store_true", help="Run a quick test training on a tiny subset")
    parser.add_argument("--render_platform", type=str, default="egl", choices=["egl", "osmesa"], help="Rendering platform (egl or osmesa)")
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
        init_renderer=False,
        map_location="cpu"
    )
    
    print("Loading a separate Teacher model to preserve original camera parameters...")
    teacher_model = HAMER.load_from_checkpoint(
        args.checkpoint,
        strict=False,
        cfg=model_cfg,
        init_renderer=False,
        map_location="cpu"
    )
    
    # Create OpenTouchHAMER model
    model = OpenTouchHAMER(
        cfg=model_cfg,
        learning_rate=lr_scaled,
        freeze_backbone=not args.no_freeze,
        teacher_model=teacher_model
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
        
    # Create DataLoaders
    # Note: PyTorch Lightning DDP automatically applies DistributedSampler, 
    # dividing datasets among active GPU worker processes.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Set up checkpoints callback
    if args.quick_test:
        ckpt_dir = os.path.join(ft_dir, "checkpoints_test")
    else:
        ckpt_dir = os.path.join(ft_dir, "checkpoints", args.exp_name) if args.exp_name else os.path.join(ft_dir, "checkpoints")
        
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
        wandb_name = args.exp_name if args.exp_name else "hamer-finetune"
        logger = WandbLogger(project="opentouch-hamer-ft", name=wandb_name)
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
