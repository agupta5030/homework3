"""
Training script for road detection task
This script trains the Detector model for segmentation and depth estimation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric


def train_epoch(model, train_loader, criterion_seg, criterion_depth, optimizer, device, lambda_seg=1.0, lambda_depth=1.0):
    """
    Train the model for one epoch
    """
    model.train()
    metric = DetectionMetric(num_classes=3)
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        # Move data to device
        images = batch["image"].to(device)
        track_labels = batch["track"].to(device)  # Segmentation labels
        depth_labels = batch["depth"].to(device)  # Depth labels
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        seg_logits, depth_pred = model(images)
        
        # Compute losses
        # Segmentation loss: cross entropy
        # Need to reshape logits from (B, C, H, W) to (B*H*W, C) and labels to (B*H*W,)
        B, C, H, W = seg_logits.shape
        seg_logits_flat = seg_logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        track_labels_flat = track_labels.view(-1)
        seg_loss = criterion_seg(seg_logits_flat, track_labels_flat)
        
        # Depth loss: L1 loss (mean absolute error)
        depth_loss = criterion_depth(depth_pred, depth_labels)
        
        # Combined loss
        total_loss_batch = lambda_seg * seg_loss + lambda_depth * depth_loss
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += total_loss_batch.item()
        num_batches += 1
        
        # Update metrics
        with torch.no_grad():
            preds = seg_logits.argmax(dim=1)
            metric.add(preds, track_labels, depth_pred, depth_labels)
    
    avg_loss = total_loss / num_batches
    metrics = metric.compute()
    
    return avg_loss, metrics


def validate(model, val_loader, criterion_seg, criterion_depth, device, lambda_seg=1.0, lambda_depth=1.0):
    """
    Validate the model on validation set
    """
    model.eval()
    metric = DetectionMetric(num_classes=3)
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.inference_mode():
        for batch in val_loader:
            images = batch["image"].to(device)
            track_labels = batch["track"].to(device)
            depth_labels = batch["depth"].to(device)
            
            # Forward pass
            seg_logits, depth_pred = model(images)
            
            # Compute losses
            B, C, H, W = seg_logits.shape
            seg_logits_flat = seg_logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
            track_labels_flat = track_labels.view(-1)
            seg_loss = criterion_seg(seg_logits_flat, track_labels_flat)
            
            depth_loss = criterion_depth(depth_pred, depth_labels)
            total_loss_batch = lambda_seg * seg_loss + lambda_depth * depth_loss
            
            # Track metrics
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Update metrics
            preds = seg_logits.argmax(dim=1)
            metric.add(preds, track_labels, depth_pred, depth_labels)
    
    avg_loss = total_loss / num_batches
    metrics = metric.compute()
    
    return avg_loss, metrics


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_loader = load_data(
        "drive_data/train",
        transform_pipeline="default",
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = load_data(
        "drive_data/val",
        transform_pipeline="default",
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    print("Creating model...")
    model = Detector(in_channels=3, num_classes=3).to(device)
    
    # Loss functions
    criterion_seg = nn.CrossEntropyLoss()
    criterion_depth = nn.L1Loss()  # Mean Absolute Error
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loss weights - you can adjust these to balance segmentation vs depth
    lambda_seg = 1.0
    lambda_depth = 1.0
    
    # Training parameters
    num_epochs = 30
    best_val_iou = 0.0
    
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion_seg, criterion_depth, optimizer, device,
            lambda_seg, lambda_depth
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion_seg, criterion_depth, device,
            lambda_seg, lambda_depth
        )
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    Train IOU: {train_metrics['iou']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"    Train Depth Error: {train_metrics['abs_depth_error']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"    Val IOU: {val_metrics['iou']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"    Val Depth Error: {val_metrics['abs_depth_error']:.4f}")
        
        # Save best model based on IoU
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            save_model(model)
            print(f"  ✓ Saved new best model (val IoU: {val_metrics['iou']:.4f})")
        
        print()
    
    print("-" * 60)
    print(f"Training complete! Best validation IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()

