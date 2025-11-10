"""
Training script for road detection model.
This script trains a model to perform both semantic segmentation and depth estimation.
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse

from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric


def train_epoch(model, train_loader, optimizer, seg_criterion, depth_criterion, device, seg_weight=1.0, depth_weight=1.0):
    """Train the model for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_seg_loss = 0.0
    total_depth_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        images = batch_data["image"].to(device)
        track_labels = batch_data["track"].to(device)
        depth_labels = batch_data["depth"].to(device)
        
        optimizer.zero_grad()
        seg_logits, depth_pred = model(images)
        
        seg_loss = seg_criterion(seg_logits, track_labels)
        depth_loss = depth_criterion(depth_pred, depth_labels)
        loss = seg_weight * seg_loss + depth_weight * depth_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_depth_loss += depth_loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Seg: {seg_loss.item():.4f}, "
                  f"Depth: {depth_loss.item():.4f}")
    
    return {
        "total_loss": total_loss / num_batches,
        "seg_loss": total_seg_loss / num_batches,
        "depth_loss": total_depth_loss / num_batches
    }


def validate(model, val_loader, seg_criterion, depth_criterion, device, seg_weight=1.0, depth_weight=1.0):
    """Evaluate the model on validation data."""
    model.eval()
    
    total_loss = 0.0
    total_seg_loss = 0.0
    total_depth_loss = 0.0
    num_batches = 0
    
    metric = DetectionMetric(num_classes=3)
    
    with torch.inference_mode():
        for batch_data in val_loader:
            images = batch_data["image"].to(device)
            track_labels = batch_data["track"].to(device)
            depth_labels = batch_data["depth"].to(device)
            
            seg_logits, depth_pred = model(images)
            
            seg_loss = seg_criterion(seg_logits, track_labels)
            depth_loss = depth_criterion(depth_pred, depth_labels)
            loss = seg_weight * seg_loss + depth_weight * depth_loss
            
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_depth_loss += depth_loss.item()
            num_batches += 1
            
            seg_pred, depth_pred = model.predict(images)
            metric.add(seg_pred, track_labels, depth_pred, depth_labels)
    
    avg_loss = total_loss / num_batches
    avg_seg_loss = total_seg_loss / num_batches
    avg_depth_loss = total_depth_loss / num_batches
    
    detection_metrics = metric.compute()
    
    return {
        "total_loss": avg_loss,
        "seg_loss": avg_seg_loss,
        "depth_loss": avg_depth_loss,
        "iou": detection_metrics["iou"],
        "accuracy": detection_metrics["accuracy"],
        "abs_depth_error": detection_metrics["abs_depth_error"],
        "tp_depth_error": detection_metrics["tp_depth_error"]
    }


def main(args):
    """Main training function."""
    print("=" * 60)
    print("Starting Road Detection Model Training")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    print("\nLoading training data...")
    train_loader = load_data(
        dataset_path=args.train_path,
        transform_pipeline="default",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    print("Loading validation data...")
    val_loader = load_data(
        dataset_path=args.val_path,
        transform_pipeline="default",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print("\nInitializing model...")
    model = Detector(in_channels=3, num_classes=3)
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    seg_criterion = nn.CrossEntropyLoss()
    depth_criterion = nn.L1Loss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    print("\n" + "=" * 60)
    print("Starting Training Loop")
    print("=" * 60)
    
    best_iou = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 60)
        
        train_losses = train_epoch(
            model, train_loader, optimizer,
            seg_criterion, depth_criterion, device,
            seg_weight=args.seg_weight,
            depth_weight=args.depth_weight
        )
        print(f"Training - Total Loss: {train_losses['total_loss']:.4f}, "
              f"Seg: {train_losses['seg_loss']:.4f}, "
              f"Depth: {train_losses['depth_loss']:.4f}")
        
        val_metrics = validate(
            model, val_loader,
            seg_criterion, depth_criterion, device,
            seg_weight=args.seg_weight,
            depth_weight=args.depth_weight
        )
        
        print(f"Validation - Total Loss: {val_metrics['total_loss']:.4f}")
        print(f"  Segmentation IoU: {val_metrics['iou']:.4f} (target: 0.75)")
        print(f"  Segmentation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Depth Error (all): {val_metrics['abs_depth_error']:.4f} (target: < 0.05)")
        print(f"  Depth Error (lanes): {val_metrics['tp_depth_error']:.4f} (target: < 0.05)")
        
        scheduler.step(val_metrics['iou'])
        
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            model_path = save_model(model)
            print(f"✓ New best model saved! IoU: {best_iou:.4f}")
            print(f"  Model saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Validation IoU: {best_iou:.4f}")
    print(f"\nTarget Metrics:")
    print(f"  - Segmentation IoU: > 0.75")
    print(f"  - Depth Error: < 0.05")
    print(f"  - Lane Depth Error: < 0.05")
    
    if best_iou >= 0.75:
        print("\n✓ IoU target achieved!")
    else:
        print("\n✗ IoU target not reached. Consider:")
        print("  - Training for more epochs")
        print("  - Adjusting loss weights (--seg_weight, --depth_weight)")
        print("  - Adjusting learning rate")
        print("  - Modifying model architecture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train detection model")
    
    parser.add_argument("--train_path", type=str, default="drive_data/train")
    parser.add_argument("--val_path", type=str, default="drive_data/val")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    
    args = parser.parse_args()
    main(args)

