#!/usr/bin/env python3
"""
Training script for HSDVC system.
Stage 1: Pre-train motion extraction and identity encoder.
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm

from hsdvc.config import HSDVCConfig, TrainingConfig
from hsdvc.models.motion import MotionExtractor
from hsdvc.models.identity import IdentityEncoder
from hsdvc.data import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train HSDVC Stage 1: Motion and Identity")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--val_data_dir", type=str, default=None, help="Validation data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/stage1", help="Output directory")
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=8, help="Data loader workers")
    
    # Model
    parser.add_argument("--motion_model", type=str, default="mediapipe", help="Pose estimation model")
    parser.add_argument("--identity_encoder", type=str, default="dinov2", help="Identity encoder backbone")
    
    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="hsdvc", help="W&B project name")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save interval")
    
    return parser.parse_args()


def train_epoch(
    motion_extractor: nn.Module,
    identity_encoder: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    args
):
    """Train for one epoch."""
    motion_extractor.train()
    identity_encoder.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        videos = batch["video"].to(device)  # [B, T, C, H, W]
        identity_images = batch["identity_image"].to(device)  # [B, C, H, W]
        
        # Forward pass
        optimizer.zero_grad()
        
        # Extract motion
        motion_data = motion_extractor(videos)
        
        # Extract identity
        identity_emb = identity_encoder(identity_images)
        
        # Compute losses
        # For motion: supervised with ground truth poses/depth/flow
        motion_loss = 0.0
        
        if "poses_gt" in batch:
            poses_gt = batch["poses_gt"].to(device)
            motion_loss += nn.functional.mse_loss(
                motion_data.poses_3d,
                poses_gt
            )
        
        # For identity: triplet loss
        identity_loss = 0.0
        
        if "identity_positive" in batch and "identity_negative" in batch:
            positive_images = batch["identity_positive"].to(device)
            negative_images = batch["identity_negative"].to(device)
            
            positive_emb = identity_encoder(positive_images)
            negative_emb = identity_encoder(negative_images)
            
            identity_losses = identity_encoder.compute_loss(
                identity_emb,
                anchor=identity_emb,
                positive=positive_emb,
                negative=negative_emb
            )
            identity_loss = identity_losses["total"]
        
        # Total loss
        loss = motion_loss + identity_loss
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(motion_extractor.parameters()) + list(identity_encoder.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        # Update stats
        total_loss += loss.item()
        num_batches += 1
        
        # Log
        if batch_idx % args.log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/motion_loss", motion_loss.item() if isinstance(motion_loss, torch.Tensor) else motion_loss, global_step)
            writer.add_scalar("train/identity_loss", identity_loss.item() if isinstance(identity_loss, torch.Tensor) else identity_loss, global_step)
            
            if args.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/motion_loss": motion_loss.item() if isinstance(motion_loss, torch.Tensor) else motion_loss,
                    "train/identity_loss": identity_loss.item() if isinstance(identity_loss, torch.Tensor) else identity_loss,
                    "global_step": global_step,
                })
        
        pbar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    motion_extractor: nn.Module,
    identity_encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    args
):
    """Validate model."""
    motion_extractor.eval()
    identity_encoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            videos = batch["video"].to(device)
            identity_images = batch["identity_image"].to(device)
            
            # Forward pass
            motion_data = motion_extractor(videos)
            identity_emb = identity_encoder(identity_images)
            
            # Compute loss (simplified)
            loss = 0.0
            
            if "poses_gt" in batch:
                poses_gt = batch["poses_gt"].to(device)
                loss += nn.functional.mse_loss(motion_data.poses_3d, poses_gt)
            
            total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    writer.add_scalar("val/loss", avg_loss, epoch)
    
    if args.use_wandb:
        wandb.log({"val/loss": avg_loss, "epoch": epoch})
    
    return avg_loss


def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # Initialize tensorboard
    writer = SummaryWriter(output_dir / "logs")
    
    # Create config
    config = HSDVCConfig()
    config.motion.pose_model = args.motion_model
    config.identity.encoder_type = args.identity_encoder
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    
    # Create models
    print("Initializing models...")
    motion_extractor = MotionExtractor(config.motion).to(device)
    identity_encoder = IdentityEncoder(config.identity).to(device)
    
    # Create optimizer
    optimizer = AdamW(
        list(motion_extractor.parameters()) + list(identity_encoder.parameters()),
        lr=args.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    val_loader = None
    if args.val_data_dir:
        val_loader = create_dataloader(
            args.val_data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
        )
    
    # Training loop
    print(f"Starting training for {args.num_epochs} epochs...")
    
    best_val_loss = float("inf")
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            motion_extractor,
            identity_encoder,
            train_loader,
            optimizer,
            device,
            epoch,
            writer,
            args
        )
        
        print(f"Train loss: {train_loss:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss = validate(
                motion_extractor,
                identity_encoder,
                val_loader,
                device,
                epoch,
                writer,
                args
            )
            print(f"Val loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best model! Saving...")
                torch.save({
                    "epoch": epoch,
                    "motion_extractor": motion_extractor.state_dict(),
                    "identity_encoder": identity_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, output_dir / "best_model.pt")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "motion_extractor": motion_extractor.state_dict(),
                "identity_encoder": identity_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")
    
    print("\nTraining complete!")
    writer.close()
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
