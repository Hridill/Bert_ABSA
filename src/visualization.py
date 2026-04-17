import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from logger import setup_logger

logger = setup_logger('visualization')

class ModelVisualizer:
    def __init__(self, output_dir='visualizations'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_training_history(self, history, save_name='training_history.png'):
        """
        Plot training metrics history
        """
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['train_accuracy'], label='Train')
        plt.plot(history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved training history plot to {save_path}")

    def plot_confusion_matrix(self, y_true, y_pred, save_name='confusion_matrix.png'):
        """
        Plot confusion matrix for three classes
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Neutral', 'Positive'],
                   yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {save_path}")

    def plot_class_distribution(self, y_true, y_pred, save_name='class_distribution.png'):
        """
        Plot distribution of predictions across classes
        """
        plt.figure(figsize=(12, 5))
        
        # Plot true distribution
        plt.subplot(1, 2, 1)
        true_counts = pd.Series(y_true).value_counts().sort_index()
        plt.bar(['Negative', 'Neutral', 'Positive'], true_counts)
        plt.title('True Class Distribution')
        plt.ylabel('Count')
        
        # Plot predicted distribution
        plt.subplot(1, 2, 2)
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        plt.bar(['Negative', 'Neutral', 'Positive'], pred_counts)
        plt.title('Predicted Class Distribution')
        plt.ylabel('Count')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved class distribution plot to {save_path}")

    def plot_class_metrics(self, y_true, y_pred, save_name='class_metrics.png'):
        """
        Plot precision, recall, and F1 score for each class
        """
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics_df = pd.DataFrame(report).transpose()
        
        plt.figure(figsize=(10, 6))
        metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
        plt.title('Class-wise Metrics')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved class metrics plot to {save_path}") 