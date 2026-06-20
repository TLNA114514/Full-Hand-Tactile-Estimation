import sys
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # Avoid HDF5 locking issues with multiprocessing
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

from src.data import EgoDexDatasetV2, get_transforms, build_dataset
from src.data.egodex_dataset_v2 import collate_fn
from src.models import build_model
from src.losses import PoseLoss, TactileLoss
from src.utils import load_config_with_base, setup_logger, compute_all_metrics, save_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train Touch Anything Model')
    parser.add_argument('--config', type=str, default='configs/hamer_pose_training_330_full.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of processes for distributed training')
    return parser.parse_args()


def set_seed(seed: int, benchmark: bool = True):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = not benchmark
    torch.backends.cudnn.benchmark = benchmark


def setup_ddp(local_rank: int):
    """Initialize distributed training."""
    if local_rank == -1:
        return False, 0, 1  # Not using DDP
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return True, rank, world_size


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def apply_pose_valid_policy(
    poses: torch.Tensor,
    confidences: torch.Tensor,
    task: str,
    config: dict,
    sensor_mask: torch.Tensor = None,
    pressure_maps: torch.Tensor = None,
):
    """
    Apply invalid-pose policy before loss computation.

    Policies:
      - keep:       do nothing (backward compatible)
      - skip_frame: mask invalid frames in loss
      - skip_clip:  mask entire clips whose valid-frame ratio is below threshold
    """
    train_cfg = config.get('training', {})
    policy = train_cfg.get('invalid_pose_policy', 'keep').lower()
    min_valid_ratio = float(train_cfg.get('invalid_pose_min_valid_ratio', 0.2))
    joint_ratio_thr = float(train_cfg.get('pose_valid_joint_ratio_threshold', 1.0))

    if policy not in {'keep', 'skip_frame', 'skip_clip'}:
        raise ValueError(f"Unknown invalid_pose_policy: {policy}")

    # Frame is valid if enough joints are valid (default: all joints valid)
    frame_joint_ratio = confidences.mean(dim=-1)  # (B, T)
    frame_valid = (frame_joint_ratio >= joint_ratio_thr).float()  # (B, T)
    clip_valid_ratio = frame_valid.mean(dim=1)  # (B,)
    clip_keep = torch.ones_like(clip_valid_ratio)

    if policy == 'skip_clip':
        clip_keep = (clip_valid_ratio >= min_valid_ratio).float()
        frame_valid = frame_valid * clip_keep[:, None]

    confidences_out = confidences
    poses_for_model = poses
    sensor_mask_out = sensor_mask

    if policy in {'skip_frame', 'skip_clip'}:
        frame_valid_j = frame_valid.unsqueeze(-1)  # (B, T, 1)
        confidences_out = confidences * frame_valid_j

        # For tactile prediction, mask invalid frames in loss computation only
        # DO NOT zero out poses - this causes NaN in normalization (0-50)/50 = -1
        if task == 'tactile_prediction':
            # Keep poses_for_model unchanged - let model see all poses
            # Only mask out invalid frames in loss via sensor_mask
            if pressure_maps is not None:
                if sensor_mask_out is None:
                    sensor_mask_out = torch.ones_like(pressure_maps)
                sensor_mask_out = sensor_mask_out * frame_valid[:, :, None, None, None]

    skip_batch = False
    if task == 'pose_prediction':
        skip_batch = bool(confidences_out.sum().item() == 0)
    elif task == 'tactile_prediction' and sensor_mask_out is not None:
        skip_batch = bool(sensor_mask_out.sum().item() == 0)

    stats = {
        'policy': policy,
        'frame_valid_ratio': float(frame_valid.mean().item()),
        'clip_keep_ratio': float(clip_keep.mean().item()),
    }

    return poses_for_model, confidences_out, sensor_mask_out, stats, skip_batch


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, logger, config):
    """Train for one epoch."""
    model.train()
    task = config['model']['task']
    
    total_loss = 0.0
    log_interval = config['training']['log_interval']
    
    multi_view = config['model'].get('multi_view', {}).get('enabled', False)

    processed_batches = 0
    skipped_batches = 0
    sum_frame_valid_ratio = 0.0
    sum_clip_keep_ratio = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        frames = batch['frames'].to(device)  # (B, T, 3, H, W)
        poses = batch['poses'].to(device)  # (B, T, J, 3)
        confidences = batch['confidences'].to(device)  # (B, T, J)

        pressure_maps = None
        sensor_mask = None
        if task == 'tactile_prediction':
            pressure_maps = batch['pressure_maps'].to(device)  # (B, T, 2, S, S)
            sensor_mask = batch.get('sensor_mask')
            if sensor_mask is not None:
                sensor_mask = sensor_mask.to(device)

        poses_for_model, conf_for_loss, sensor_mask, pose_stats, skip_batch = apply_pose_valid_policy(
            poses=poses,
            confidences=confidences,
            task=task,
            config=config,
            sensor_mask=sensor_mask,
            pressure_maps=pressure_maps,
        )

        if skip_batch:
            skipped_batches += 1
            continue

        processed_batches += 1
        sum_frame_valid_ratio += pose_stats['frame_valid_ratio']
        sum_clip_keep_ratio += pose_stats['clip_keep_ratio']
        
        # Prepare views dict if multi-view
        views = None
        if multi_view and 'views' in batch:
            views = {k: v.to(device) for k, v in batch['views'].items()}
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config['system']['mixed_precision']:
            with autocast():
                if task == 'pose_prediction':
                    # Multi-view: pass only ``views``; single-view: pass only ``frames``.
                    if multi_view and views is not None:
                        outputs = model(views=views)
                    else:
                        outputs = model(frames=frames)
                    loss = criterion(outputs['poses'], poses, conf_for_loss)
                elif task == 'tactile_prediction':
                    # Multi-view: pass only ``views``; single-view: pass only ``frames``.
                    if multi_view and views is not None:
                        outputs = model(poses=poses_for_model, views=views)
                    else:
                        outputs = model(frames=frames, poses=poses_for_model)
                    
                    # # DEBUG: NaN diagnostics (commented out for production training)
                    # if logger is not None:
                    #     if torch.isnan(poses_for_model).any():
                    #         logger.error(f"[NaN] poses_for_model contains NaN!")
                    #     else:
                    #         logger.info(f"[DEBUG] poses_for_model range: [{poses_for_model.min():.4f}, {poses_for_model.max():.4f}]")
                    #         logger.info(f"[DEBUG] poses_for_model mean: {poses_for_model.mean():.4f}, std: {poses_for_model.std():.4f}")
                    #     
                    #     if torch.isnan(pressure_maps).any():
                    #         logger.error(f"[NaN] pressure_maps contains NaN!")
                    #         logger.error(f"  NaN count: {torch.isnan(pressure_maps).sum().item()}")
                    #     else:
                    #         logger.info(f"[DEBUG] pressure_maps range: [{pressure_maps.min():.4f}, {pressure_maps.max():.4f}]")
                    #     
                    #     if sensor_mask is not None:
                    #         if torch.isnan(sensor_mask).any():
                    #             logger.error(f"[NaN] sensor_mask contains NaN!")
                    #         valid_ratio = sensor_mask.sum() / sensor_mask.numel()
                    #         logger.info(f"[DEBUG] sensor_mask: {sensor_mask.sum().item()}/{sensor_mask.numel()} valid ({valid_ratio:.2%})")
                    #     
                    #     if torch.isnan(outputs['tactile']).any():
                    #         logger.error(f"[NaN] model output contains NaN!")
                    #         logger.error(f"  Output range: [{outputs['tactile'].min():.4f}, {outputs['tactile'].max():.4f}]")
                    
                    loss = criterion(outputs['tactile'], pressure_maps, mask=sensor_mask)
                    
                    # # DEBUG: NaN loss detection (commented out for production training)
                    # if torch.isnan(loss):
                    #     if logger is not None:
                    #         logger.error(f"[NaN] Loss is NaN after criterion!")
                    #         logger.error(f"  pred range: [{outputs['tactile'].min():.4f}, {outputs['tactile'].max():.4f}]")
                    #         logger.error(f"  target range: [{pressure_maps.min():.4f}, {pressure_maps.max():.4f}]")
                    #         logger.error(f"  mask sum: {sensor_mask.sum()}")
                    #         logger.error(f"  poses range: [{poses_for_model.min():.4f}, {poses_for_model.max():.4f}]")
                    #     raise ValueError("NaN loss detected!")
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if task == 'pose_prediction':
                # Multi-view: pass only ``views``; single-view: pass only ``frames``.
                if multi_view and views is not None:
                    outputs = model(views=views)
                else:
                    outputs = model(frames=frames)
                loss = criterion(outputs['poses'], poses, conf_for_loss)
            elif task == 'tactile_prediction':
                # Multi-view: pass only ``views``; single-view: pass only ``frames``.
                if multi_view and views is not None:
                    outputs = model(poses=poses_for_model, views=views)
                else:
                    outputs = model(frames=frames, poses=poses_for_model)
                loss = criterion(outputs['tactile'], pressure_maps, mask=sensor_mask)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            denom = max(processed_batches, 1)
            avg_loss = total_loss / denom
            avg_valid_ratio = sum_frame_valid_ratio / denom
            avg_clip_keep = sum_clip_keep_ratio / denom
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            if logger is not None:
                logger.info(
                    f'Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] '
                    f'Loss: {avg_loss:.4f} | pose_valid_ratio: {avg_valid_ratio:.3f} '
                    f'| clip_keep_ratio: {avg_clip_keep:.3f}'
                )
    
    if processed_batches == 0:
        if logger is not None:
            logger.warning(
                f'Epoch {epoch}: no valid batches after invalid-pose filtering '
                f'(skipped={skipped_batches})'
            )
        return 0.0

    avg_loss = total_loss / processed_batches
    if logger is not None:
        logger.info(
            f'Epoch {epoch} summary | policy={config.get("training", {}).get("invalid_pose_policy", "keep")} '
            f'| processed={processed_batches} skipped={skipped_batches} '
            f'| pose_valid_ratio={sum_frame_valid_ratio/processed_batches:.3f} '
        f'| clip_keep_ratio={sum_clip_keep_ratio/processed_batches:.3f}'
    )
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, logger, config):
    """Validate the model."""
    model.eval()
    task = config['model']['task']
    
    total_loss = 0.0
    all_metrics = {'mpjpe': [], 'pa_mpjpe': [], 'pck': []} if task == 'pose_prediction' else {}
    
    multi_view = config['model'].get('multi_view', {}).get('enabled', False)

    processed_batches = 0
    skipped_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Validation')
    for batch in pbar:
        frames = batch['frames'].to(device)
        poses = batch['poses'].to(device)
        confidences = batch['confidences'].to(device)

        pressure_maps = None
        sensor_mask = None
        if task == 'tactile_prediction':
            pressure_maps = batch['pressure_maps'].to(device)
            sensor_mask = batch.get('sensor_mask')
            if sensor_mask is not None:
                sensor_mask = sensor_mask.to(device)

        poses_for_model, conf_for_loss, sensor_mask, _, skip_batch = apply_pose_valid_policy(
            poses=poses,
            confidences=confidences,
            task=task,
            config=config,
            sensor_mask=sensor_mask,
            pressure_maps=pressure_maps,
        )
        if skip_batch:
            skipped_batches += 1
            continue

        processed_batches += 1
        
        # Prepare views dict if multi-view
        views = None
        if multi_view and 'views' in batch:
            views = {k: v.to(device) for k, v in batch['views'].items()}
        
        if task == 'pose_prediction':
            # Multi-view: pass only ``views``; single-view: pass only ``frames``.
            if multi_view and views is not None:
                outputs = model(views=views)
            else:
                outputs = model(frames=frames)
            pred_poses = outputs['poses']
            loss = criterion(pred_poses, poses, conf_for_loss)
            total_loss += loss.item()
            
            metrics = compute_all_metrics(
                pred_poses, poses, conf_for_loss,
                threshold=config['evaluation']['pck_threshold']
            )
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        elif task == 'tactile_prediction':
            # Multi-view: pass only ``views``; single-view: pass only ``frames``.
            if multi_view and views is not None:
                outputs = model(poses=poses_for_model, views=views)
            else:
                outputs = model(frames=frames, poses=poses_for_model)
            loss = criterion(outputs['tactile'], pressure_maps, mask=sensor_mask)
            total_loss += loss.item()
    
    # Average metrics
    if processed_batches == 0:
        if logger is not None:
            logger.warning(
                f'Validation Epoch {epoch}: no valid batches after invalid-pose filtering '
                f'(skipped={skipped_batches})'
            )
        return 0.0, {'val_loss': 0.0}

    avg_loss = total_loss / processed_batches
    
    if task == 'pose_prediction':
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        if logger is not None:
            logger.info(f'Validation Epoch {epoch}:')
            logger.info(f'  Loss: {avg_loss:.4f}')
            logger.info(f'  MPJPE: {avg_metrics["mpjpe"]*1000:.2f} mm')
            logger.info(f'  PA-MPJPE: {avg_metrics["pa_mpjpe"]*1000:.2f} mm')
            logger.info(f'  PCK@{config["evaluation"]["pck_threshold"]*100:.0f}cm: {avg_metrics["pck"]*100:.2f}%')
    else:
        avg_metrics = {'val_loss': avg_loss}
        if logger is not None:
            logger.info(f'Validation Epoch {epoch}:')
            logger.info(f'  Loss: {avg_loss:.6f}')
    
    return avg_loss, avg_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
        'metrics': metrics,
    }
    torch.save(checkpoint, save_path)


def main():
    args = parse_args()
    
    # Get local_rank from environment variable (set by torchrun)
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    
    # Setup DDP
    use_ddp, rank, world_size = setup_ddp(local_rank)
    is_main_process = (rank == 0)
    
    # Load config
    config = load_config_with_base(args.config)
    
    # Setup logger (only on main process)
    if is_main_process:
        logger = setup_logger(name='Train', log_dir=config.get('log_dir', 'outputs/logs'))
        logger.info(f'Config: {args.config}')
    else:
        logger = None
    
    # Set seed
    set_seed(config['system']['seed'] + rank, benchmark=config['system'].get('cudnn_benchmark', True))
    
    # Setup device
    if use_ddp:
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process:
            logger.info(f'Using DDP with {world_size} processes, local_rank={local_rank}')
    else:
        explicit_device = args.device not in {'cuda', 'cpu'}
        if explicit_device:
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        elif 'gpu_ids' in config['system'] and len(config['system']['gpu_ids']) > 0 and torch.cuda.is_available():
            gpu_ids = config['system']['gpu_ids']
            device = torch.device(f'cuda:{gpu_ids[0]}')
            if is_main_process:
                logger.info(f'Using configured single GPU: {gpu_ids[0]} (gpu_ids={gpu_ids})')
        else:
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if is_main_process:
        logger.info(f'Using device: {device}')
    
    # Create datasets
    if is_main_process:
        logger.info('Creating datasets...')
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    # Create datasets using factory (supports egodex_v2 / touchanything)
    dataset_type = config['data'].get('dataset', 'egodex_v2')
    if is_main_process:
        logger.info(f'Dataset type: {dataset_type}')
    train_dataset = build_dataset(config, split='train', transform=train_transform)
    val_dataset   = build_dataset(config, split='val',   transform=val_transform)
    
    # Create dataloaders with DistributedSampler for DDP
    num_workers = config['training']['num_workers']
    
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config['system']['seed'],
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        shuffle_train = False
        shuffle_val = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True
        shuffle_val = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle_val,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    if is_main_process:
        logger.info(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    # Build model
    if is_main_process:
        logger.info('Building model...')
    model = build_model(config)
    model = model.to(device)
    
    # Wrap model with DDP
    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # Allow unused parameters (e.g. some multi-task branches).
        )
        if is_main_process:
            logger.info(f'Model wrapped with DDP on device {local_rank}')
    else:
        if is_main_process:
            logger.info('Using single GPU')
    
    # Loss function
    task = config['model']['task']
    if task == 'pose_prediction':
        criterion = PoseLoss(
            use_confidence_weighting=config['training']['loss']['use_confidence_weighting']
        )
    elif task == 'tactile_prediction':
        loss_cfg = config['training']['loss']
        criterion = TactileLoss(
            mse_weight=loss_cfg.get('mse_weight', 1.0),
            l1_weight=loss_cfg.get('l1_weight', 0.5),
            tv_weight=loss_cfg.get('tv_weight', 0.01),
            contact_weight=loss_cfg.get('contact_weight', 2.0),
            pressure_threshold=loss_cfg.get('pressure_threshold', 0.1),
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Optimizer
    optimizer_cfg = config['training']['optimizer']
    if optimizer_cfg['type'].lower() == 'adamw':
        # Only optimize trainable parameters (skip frozen encoder)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if is_main_process:
            logger.info(f'Optimizing {len(trainable_params)} parameter groups '
                        f'({sum(p.numel() for p in trainable_params):,} params)')
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=optimizer_cfg['lr'],
            weight_decay=optimizer_cfg['weight_decay'],
            betas=optimizer_cfg['betas'],
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_cfg['type']}")
    
    # Scheduler with warmup
    scheduler_cfg = config['training']['scheduler']
    warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)
    
    if scheduler_cfg['type'].lower() == 'cosine':
        # Cosine scheduler (applied after warmup)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'] - warmup_epochs,
            eta_min=scheduler_cfg['min_lr'],
        )
        
        if warmup_epochs > 0:
            # Linear warmup scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
            if is_main_process:
                logger.info(f'Using CosineAnnealingLR with {warmup_epochs} warmup epochs')
        else:
            scheduler = cosine_scheduler
            if is_main_process:
                logger.info(f'Using CosineAnnealingLR without warmup')

    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = GradScaler() if config['system']['mixed_precision'] else None
    
    # Create experiment directory with timestamp and model name (only on main process)
    if is_main_process:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = config['model']['vision_encoder']['model_name']  # e.g., dinov2_vits14
        task = config['model']['task']  # e.g., pose_prediction
        
        # Extract config filename without extension
        config_filename = Path(args.config).stem  # e.g., touchanything_with_glove_aug_wilor
        exp_name = f"{timestamp}_{model_name}_{task}_{config_filename}"
        
        checkpoint_dir = Path('checkpoints') / exp_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f'Experiment directory: {checkpoint_dir}')
        
        # Save config
        config_save_path = checkpoint_dir / 'config.yaml'
        save_config(config, str(config_save_path))
        logger.info(f'Config saved to {config_save_path}')
    
    # Broadcast checkpoint_dir to all processes
    if use_ddp:
        if is_main_process:
            checkpoint_dir_str = str(checkpoint_dir)
        else:
            checkpoint_dir_str = None
        # Simple broadcast using object list
        checkpoint_dir_list = [checkpoint_dir_str]
        dist.broadcast_object_list(checkpoint_dir_list, src=0)
        if not is_main_process:
            checkpoint_dir = Path(checkpoint_dir_list[0])
    else:
        pass  # checkpoint_dir already set
    
    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        if is_main_process:
            logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Handle module. prefix mismatch (DataParallel <-> single GPU)
        state_dict = checkpoint['model_state_dict']
        model_is_dp = isinstance(model, nn.DataParallel)
        ckpt_is_dp = any(k.startswith('module.') for k in state_dict.keys())
        
        if ckpt_is_dp and not model_is_dp:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not ckpt_is_dp and model_is_dp:
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if is_main_process:
                logger.info(f'Scheduler state restored')
        
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    if is_main_process:
        logger.info('Starting training...')
    best_metric = float('inf')  # lower is better (MPJPE for pose, loss for tactile)
    
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        # Set epoch for DistributedSampler
        if use_ddp:
            train_sampler.set_epoch(epoch)
        
        if is_main_process:
            logger.info(f'\n{"="*60}')
            logger.info(f'Epoch {epoch}/{config["training"]["num_epochs"]}')
            logger.info(f'{"="*60}')
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, logger if is_main_process else None, config
        )
        
        # Validate
        if epoch % config['training']['val_interval'] == 0:
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, epoch, logger if is_main_process else None, config
            )
            
            # Only save checkpoints on main process
            if is_main_process:
                # Determine tracking metric
                if task == 'pose_prediction':
                    current_metric = val_metrics['mpjpe']
                    metric_str = f'MPJPE: {current_metric*1000:.2f} mm'
                else:
                    current_metric = val_loss
                    metric_str = f'Loss: {current_metric:.6f}'
                
                # Save best model
                if current_metric < best_metric:
                    best_metric = current_metric
                    save_path = checkpoint_dir / 'best_model.pth'
                    # Extract model state_dict properly for DDP
                    model_to_save = model.module if use_ddp else model
                    save_checkpoint(model_to_save, optimizer, scheduler, epoch, val_loss, val_metrics, save_path)
                    logger.info(f'Best model saved to {save_path} ({metric_str})')
        
        # Save checkpoint
        if is_main_process and epoch % config['training']['save_interval'] == 0:
            save_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            model_to_save = model.module if use_ddp else model
            save_checkpoint(model_to_save, optimizer, scheduler, epoch, train_loss, {}, save_path)
            logger.info(f'Checkpoint saved to {save_path}')
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
            if is_main_process:
                logger.info(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')
    
    if is_main_process:
        logger.info('\nTraining completed!')
        if task == 'pose_prediction':
            logger.info(f'Best MPJPE: {best_metric*1000:.2f} mm')
        else:
            logger.info(f'Best Val Loss: {best_metric:.6f}')
    
    # Cleanup DDP
    cleanup_ddp()


if __name__ == '__main__':
    main()
