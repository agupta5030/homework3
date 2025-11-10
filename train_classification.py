"""
Training script for image classification model.
This script trains a CNN to classify SuperTuxKart game images into 6 classes.
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse

from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, device):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    metric = AccuracyMetric()
    
    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
            
            predictions = model.predict(images)
            metric.add(predictions, labels)
    
    avg_loss = total_loss / num_batches
    metrics = metric.compute()
    accuracy = metrics["accuracy"]
    
    return avg_loss, accuracy


def main(args):
    """Main training function."""
    print("=" * 60)
    print("Starting Classification Model Training")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    print("\nLoading training data...")
    train_loader = load_data(
        dataset_path=args.train_path,
        transform_pipeline="aug",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    
    print("Loading validation data...")
    val_loader = load_data(
        dataset_path=args.val_path,
        transform_pipeline="default",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    print("\nInitializing model...")
    model = Classifier(in_channels=3, num_classes=6)
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    print("\n" + "=" * 60)
    print("Starting Training Loop")
    print("=" * 60)
    
    best_accuracy = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 60)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        scheduler.step(val_loss)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model_path = save_model(model)
            print(f"✓ New best model saved! Accuracy: {best_accuracy:.4f}")
            print(f"  Model saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"Target Accuracy: 0.80")
    
    if best_accuracy >= 0.80:
        print("✓ Target accuracy achieved!")
    else:
        print("✗ Target accuracy not reached. Consider:")
        print("  - Training for more epochs")
        print("  - Adjusting learning rate")
        print("  - Modifying data augmentation")
        print("  - Tuning model architecture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification model")
    
    parser.add_argument("--train_path", type=str, default="classification_data/train")
    parser.add_argument("--val_path", type=str, default="classification_data/val")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=2)
    
    args = parser.parse_args()
    main(args)

