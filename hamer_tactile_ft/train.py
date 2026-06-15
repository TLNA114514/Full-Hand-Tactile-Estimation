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

sys.argv[0] = os.path.abspath(__file__)

import faulthandler
faulthandler.enable()

import argparse
import pytorch_lightning as pl
import cv2

cv2.setNumThreads(0)

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Add paths
ft_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(ft_dir, ".."))
sys.path.append(os.path.join(workspace_dir, "hamer"))
sys.path.append(ft_dir)

from dataset import OpenTouchTactileDataset
from hamer_tactile import HAMER_Tactile

class OpenTouchHAMER_TactileWrapper(HAMER_Tactile):
    def __init__(self, cfg, learning_rate=1e-4):
        # Initialize without loading rendering to save GPU memory
        super().__init__(cfg, init_renderer=False)
        self.learning_rate = learning_rate
        
        self.automatic_optimization = True
        
        print("Freezing the backbone and MANO head. Only training the tactile head...")
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Freeze mano head
        for param in self.mano_head.parameters():
            param.requires_grad = False
            
        # Keep the tactile head trainable
        for param in self.tactile_head.parameters():
            param.requires_grad = True
                
    def training_step(self, batch, batch_idx):
        output = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, output, train=True)
        
        # Log tactile loss and total loss
        self.log('train/loss_tactile', output['losses']['loss_tactile'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        
        # Log validation MAE / RMSE (using L1 loss directly as MAE for unnormalized)
        pred_tactile = output['pred_tactile']
        gt_tactile = batch['tactile_signal']
        has_tactile = batch['has_tactile']
        palm_mask = batch['palm_mask']
        
        valid_samples = has_tactile.sum()
        num_palm_vertices = palm_mask[0].sum() if palm_mask.shape[0] > 0 else 0
        if valid_samples > 0 and num_palm_vertices > 0:
            abs_err = torch.abs(pred_tactile - gt_tactile) * palm_mask
            mae = (abs_err * has_tactile.unsqueeze(1)).sum() / (valid_samples * num_palm_vertices)
            
            sq_err = ((pred_tactile - gt_tactile) ** 2) * palm_mask
            rmse = torch.sqrt((sq_err * has_tactile.unsqueeze(1)).sum() / (valid_samples * num_palm_vertices))
            
            self.log('val/mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('val/rmse', rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        self.log('val/loss_tactile', output['losses']['loss_tactile'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
        
    def tensorboard_logging(self, *args, **kwargs):
        # Override to be a no-op to prevent WebGL/OpenGL/Renderer crashes during training
        pass

    def configure_optimizers(self):
        tactile_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                tactile_params.append(param)
                    
        optimizer = torch.optim.AdamW(
            tactile_params,
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        print(f"Total training steps for LR Scheduler: {total_steps}")
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=self.learning_rate * 0.01
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
    parser = argparse.ArgumentParser(description="Fine-tune Hamer Tactile Head")
    # Default to the fine-tuned checkpoint!
    parser.add_argument("--checkpoint", type=str, default=os.path.join(workspace_dir, "opentouch_hamer_ft/checkpoints/regression_only/best_ft_model.ckpt"), help="Path to fine-tuned Hamer checkpoint")
    parser.add_argument("--data_dir", type=str, default="/data/jiangrui/OpenTouch Data/extracted_dataset", help="Data folder path")
    
    parser.add_argument("--gpus", type=str, default="4", help="GPU indices (comma-separated, e.g. 4,5)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate (per GPU)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--exp_name", type=str, default="tactile_ft", help="Experiment name")
    parser.add_argument("--quick_test", action="store_true", help="Run a quick test training")
    parser.add_argument("--render_platform", type=str, default="egl", choices=["egl", "osmesa"], help="Rendering platform (egl or osmesa)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    hamer_root = os.path.join(workspace_dir, "hamer")
    os.chdir(hamer_root)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpus = len(args.gpus.split(","))
    
    lr_scaled = args.lr
    if num_gpus > 1:
        lr_scaled = args.lr * num_gpus
        print(f"\n>>> Multi-GPU training detected. Scaling base learning rate linearly: {args.lr} * {num_gpus} = {lr_scaled}")
    
    print(f"Loading finetuned Hamer weights from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    from pathlib import Path
    from hamer.configs import get_config
    
    # We can use the original model_config.yaml from hamer
    model_cfg_path = os.path.join(workspace_dir, 'hamer/_DATA/hamer_ckpts/model_config.yaml')
    model_cfg = get_config(model_cfg_path, update_cachedir=True)
    
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
        
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()
        
    print("Initializing HAMER_Tactile Wrapper...")
    model = OpenTouchHAMER_TactileWrapper(
        cfg=model_cfg,
        learning_rate=lr_scaled
    )
    
    print("Loading weights from fine-tuned model...")
    state_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
    
    # Load strictly the matching keys, skipping tactile_head which is new
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys during load (should only be tactile_head): {len(missing)}")
    
    # Do a dummy forward pass to initialize the LazyLinear in tactile_head
    dummy_input = torch.zeros(1, 3, model_cfg.MODEL.IMAGE_SIZE, model_cfg.MODEL.IMAGE_SIZE)
    with torch.no_grad():
        dummy_feat = model.backbone(dummy_input[:, :, :, 32:-32])
        model.tactile_head(dummy_feat)
        print("LazyLinear in tactile_head has been initialized.")
        
    train_dataset = OpenTouchTactileDataset(
        cfg=model_cfg,
        split="train",
        data_dir=args.data_dir,
        train=True
    )
    
    val_dataset = OpenTouchTactileDataset(
        cfg=model_cfg,
        split="val",
        data_dir=args.data_dir,
        train=False
    )
    
    if args.quick_test:
        train_dataset.samples = train_dataset.samples[:64]
        val_dataset.samples = val_dataset.samples[:32]
        args.epochs = 1
        args.num_workers = 0
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    ckpt_dir = os.path.join(ft_dir, "checkpoints", args.exp_name) if not args.quick_test else os.path.join(ft_dir, "checkpoints_test")
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename="best_tactile_model", monitor="val/loss_tactile", mode="min", save_top_k=1, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    strategy = "ddp_find_unused_parameters_true" if num_gpus > 1 else "auto"
    
    if args.use_wandb:
        logger = WandbLogger(project="opentouch-hamer-tactile-ft", name=args.exp_name)
    else:
        logger = True
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=num_gpus,
        strategy=strategy,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"\nFine-tuning completed. Best checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
