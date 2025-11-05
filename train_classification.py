"""
Training script for classification task
This script trains the Classifier model on the SuperTuxKart classification dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    """
    model.train()
    metric = AccuracyMetric()
    
    total_loss = 0.0
    num_batches = 0
    
    for images, labels in train_loader:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update accuracy metric
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            metric.add(preds, labels)
    
    avg_loss = total_loss / num_batches
    metrics = metric.compute()
    
    return avg_loss, metrics["accuracy"]


def validate(model, val_loader, criterion, device):
    """
    Validate the model on validation set
    """
    model.eval()
    metric = AccuracyMetric()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update accuracy metric
            preds = logits.argmax(dim=1)
            metric.add(preds, labels)
    
    avg_loss = total_loss / num_batches
    metrics = metric.compute()
    
    return avg_loss, metrics["accuracy"]


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_loader = load_data(
        "classification_data/train",
        transform_pipeline="aug",  # Use augmentation for training
        batch_size=128,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = load_data(
        "classification_data/val",
        transform_pipeline="default",  # No augmentation for validation
        batch_size=128,
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    print("Creating model...")
    model = Classifier(in_channels=3, num_classes=6).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 20
    best_val_acc = 0.0
    
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model)
            print(f"  ✓ Saved new best model (val acc: {val_acc:.4f})")
        
        print()
    
    print("-" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

