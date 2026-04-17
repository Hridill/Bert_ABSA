import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from visualization import ModelVisualizer
import json

def generate_sample_data():
    """Generate sample data for visualizations"""
    # Sample training history
    history = {
        'train_accuracy': [0.65, 0.72, 0.78, 0.82, 0.85],
        'val_accuracy': [0.63, 0.70, 0.75, 0.80, 0.83],
        'train_loss': [0.8, 0.6, 0.5, 0.4, 0.3],
        'val_loss': [0.85, 0.65, 0.55, 0.45, 0.35]
    }
    
    # Sample predictions and true labels
    y_true = np.random.choice([0, 1, 2], size=1000, p=[0.3, 0.4, 0.3])
    y_pred = np.random.choice([0, 1, 2], size=1000, p=[0.25, 0.45, 0.3])
    
    return history, y_true, y_pred

def main():
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Initialize visualizer
    visualizer = ModelVisualizer(output_dir='visualizations')
    
    # Generate sample data
    history, y_true, y_pred = generate_sample_data()
    
    # Generate visualizations
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(y_true, y_pred)
    visualizer.plot_class_distribution(y_true, y_pred)
    
    # Generate sample metrics
    metrics = {
        'accuracy': 0.85,
        'f1_score': 0.84,
        'precision': 0.83,
        'recall': 0.82
    }
    
    # Create model_versions directory if it doesn't exist
    os.makedirs('model_versions', exist_ok=True)
    
    # Save metrics
    with open('model_versions/latest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("Sample visualizations and metrics generated successfully!")

if __name__ == "__main__":
    main() 