import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import shutil
from datetime import datetime
import torch
from logger import setup_logger

logger = setup_logger('model_versioning')

class ModelVersioning:
    def __init__(self, model_dir='model_versions'):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def save_model_version(self, model, metrics, model_name='bert_sentiment'):
        """
        Save model version with metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = os.path.join(self.model_dir, f"{model_name}_{timestamp}")
        os.makedirs(version_dir)

        # Save model
        model_path = os.path.join(version_dir, 'model.bin')
        torch.save(model.state_dict(), model_path)

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'metrics': metrics,
            'model_name': model_name
        }
        metadata_path = os.path.join(version_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Saved model version {timestamp} with metrics: {metrics}")
        return version_dir

    def load_model_version(self, version_dir, model):
        """
        Load specific model version
        """
        model_path = os.path.join(version_dir, 'model.bin')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        model.load_state_dict(torch.load(model_path))
        logger.info(f"Loaded model from {version_dir}")
        return model

    def get_version_metrics(self, version_dir):
        """
        Get metrics for specific version
        """
        metadata_path = os.path.join(version_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata['metrics']

    def list_versions(self):
        """
        List all available model versions
        """
        versions = []
        for version_dir in os.listdir(self.model_dir):
            full_path = os.path.join(self.model_dir, version_dir)
            if os.path.isdir(full_path):
                metadata_path = os.path.join(full_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    versions.append({
                        'version': version_dir,
                        'timestamp': metadata['timestamp'],
                        'metrics': metadata['metrics']
                    })
        return sorted(versions, key=lambda x: x['timestamp'], reverse=True) 