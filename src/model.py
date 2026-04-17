import torch
import torch.nn as nn
from transformers import BertModel
from config import BERT_PATH


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        
        # Freeze BERT layers except the last few
        for param in list(self.bert.parameters())[:-4]:
            param.requires_grad = False
            
        # Enhanced architecture
        self.drop1 = nn.Dropout(0.3)
        
        # Project concatenated hidden states back to 768 dimensions
        self.hidden_projection = nn.Linear(3072, 768)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(768, num_heads=8)
        self.norm1 = nn.LayerNorm(768)
        
        # Additional layers
        self.fc1 = nn.Linear(768, 512)
        self.activation = nn.GELU()
        self.drop2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)
        self.out = nn.Linear(256, 3)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training"""
        for module in [self.hidden_projection, self.fc1, self.fc2, self.out]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, mask, token_type_ids):
        # Ensure inputs are tensors
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)
        if not isinstance(token_type_ids, torch.Tensor):
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

        # Move tensors to the same device as the model
        device = next(self.parameters()).device
        ids = ids.to(device)
        mask = mask.to(device)
        token_type_ids = token_type_ids.to(device)

        # Get BERT output
        outputs = self.bert(
            ids, 
            attention_mask=mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        # Get all hidden states
        hidden_states = outputs.hidden_states
        
        # Concatenate last 4 hidden states
        last_four_hidden = torch.cat([hidden_states[-1], hidden_states[-2], 
                                    hidden_states[-3], hidden_states[-4]], dim=-1)
        
        # Project concatenated hidden states back to 768 dimensions
        projected_hidden = self.hidden_projection(last_four_hidden)
        
        # Apply self-attention
        attn_output, _ = self.attention(projected_hidden, projected_hidden, projected_hidden)
        attn_output = self.norm1(attn_output + projected_hidden)
        
        # Get pooled output
        pooled = attn_output[:, 0]  # Use [CLS] token
        
        # Apply enhanced architecture
        x = self.drop1(pooled)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.activation(x)
        output = self.out(x)
        
        return output
