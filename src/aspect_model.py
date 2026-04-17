import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class AspectClassifier(nn.Module):
    """Small transformer-based classifier for aspect prediction."""
    def __init__(self, pretrained_model_name: str = 'distilbert-base-uncased', num_labels: int = 5):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token or mean pool depending on model
        if hasattr(outputs, 'last_hidden_state'):
            cls = outputs.last_hidden_state[:, 0]
        else:
            cls = outputs[0][:, 0]
        x = self.dropout(cls)
        return self.output(x)

