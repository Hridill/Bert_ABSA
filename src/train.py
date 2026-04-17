import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW, get_cosine_schedule_with_warmup
from model import BERTBaseUncased
from logger import setup_logger
from model_versioning import ModelVersioning
from visualization import ModelVisualizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

logger = setup_logger('training')
model_versioning = ModelVersioning()
visualizer = ModelVisualizer()

def run():
    logger.info("Starting training process")
    
    # Load and preprocess data
    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")
    # Convert sentiment to numeric labels (0: negative, 1: neutral, 2: positive)
    sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
    dfx.sentiment = dfx.sentiment.map(sentiment_map)
    logger.info(f"Loaded dataset with {len(dfx)} samples")

    # Split data with stratification
    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.sentiment.values
    )
    logger.info(f"Split data into {len(df_train)} training and {len(df_valid)} validation samples")

    # Create datasets with data augmentation
    train_dataset = dataset.BERTDataset(
        review=df_train.review.values, 
        target=df_train.sentiment.values,
        augment=True
    )
    valid_dataset = dataset.BERTDataset(
        review=df_valid.review.values, 
        target=df_valid.sentiment.values,
        augment=False
    )

    # Create data loaders with weighted sampling
    class_weights = 1.0 / torch.tensor([
        (df_train.sentiment == i).sum() for i in range(3)
    ])
    class_weights = class_weights / class_weights.sum()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=class_weights[df_train.sentiment.values],
        num_samples=len(df_train),
        replacement=True
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.TRAIN_BATCH_SIZE,
        sampler=sampler,
        num_workers=4
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    # Initialize model and move to device
    device = torch.device(config.DEVICE)
    model = BERTBaseUncased()
    model.to(device)
    logger.info(f"Model initialized and moved to {device}")

    # Setup optimizer with weight decay
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Calculate total steps for scheduler
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    
    # Use OneCycleLR scheduler
    optimizer = AdamW(optimizer_parameters, lr=2e-5)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=2e-5,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_data_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    best_accuracy = 0
    history = {
        'train_accuracy': [],
        'val_accuracy': [],
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(config.EPOCHS):
        logger.info(f"Starting epoch {epoch + 1}/{config.EPOCHS}")
        
        # Train with mixed precision
        train_loss, train_accuracy = engine.train_fn(
            train_data_loader, model, optimizer, device, scheduler, scaler
        )
        
        # Validate
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        predictions = np.argmax(outputs, axis=1)
        val_accuracy = metrics.accuracy_score(targets, predictions)
        
        # Calculate validation loss
        val_loss = nn.CrossEntropyLoss()(
            torch.tensor(outputs, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.long)
        ).item()
        
        # Update history
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        logger.info(f"Epoch {epoch + 1} - Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            metrics_dict = {
                'accuracy': val_accuracy,
                'f1_score': metrics.f1_score(targets, predictions, average='weighted'),
                'precision': metrics.precision_score(targets, predictions, average='weighted'),
                'recall': metrics.recall_score(targets, predictions, average='weighted'),
                'class_metrics': {
                    'negative': {
                        'precision': metrics.precision_score(targets, predictions, labels=[0], average='micro'),
                        'recall': metrics.recall_score(targets, predictions, labels=[0], average='micro'),
                        'f1': metrics.f1_score(targets, predictions, labels=[0], average='micro')
                    },
                    'neutral': {
                        'precision': metrics.precision_score(targets, predictions, labels=[1], average='micro'),
                        'recall': metrics.recall_score(targets, predictions, labels=[1], average='micro'),
                        'f1': metrics.f1_score(targets, predictions, labels=[1], average='micro')
                    },
                    'positive': {
                        'precision': metrics.precision_score(targets, predictions, labels=[2], average='micro'),
                        'recall': metrics.recall_score(targets, predictions, labels=[2], average='micro'),
                        'f1': metrics.f1_score(targets, predictions, labels=[2], average='micro')
                    }
                }
            }
            
            # Save model version
            model_versioning.save_model_version(model, metrics_dict)
            
            # Save model for app use
            torch.save(model.state_dict(), config.MODEL_PATH)
            
            # Generate visualizations
            visualizer.plot_training_history(history)
            visualizer.plot_confusion_matrix(targets, predictions)
            visualizer.plot_class_distribution(targets, predictions)
            
            logger.info(f"New best model saved with accuracy: {val_accuracy:.4f}")

    logger.info("Training completed")
    return model, history

if __name__ == "__main__":
    run()