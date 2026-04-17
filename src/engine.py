import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from tqdm import tqdm
from logger import setup_logger
from torch.cuda.amp import autocast
import numpy as np

logger = setup_logger('engine')

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler, scaler):
    model.train()
    fin_loss = 0
    fin_preds = []
    fin_targets = []
    
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        sentiment = d["sentiment"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        sentiment = sentiment.to(device, dtype=torch.long)

        optimizer.zero_grad()
        
        # Use mixed precision training
        with autocast():
            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = nn.CrossEntropyLoss()(outputs, sentiment)
        
        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        fin_loss += loss.item()
        fin_preds.extend(torch.argmax(outputs, axis=1).cpu().detach().numpy())
        fin_targets.extend(sentiment.cpu().detach().numpy())
        
        tk0.set_postfix(loss=loss.item())

    return fin_loss / len(data_loader), sum(1 for x, y in zip(fin_preds, fin_targets) if x == y) / len(fin_preds)


def eval_fn(data_loader, model, device):
    model.eval()
    fin_outputs = []
    fin_targets = []
    
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            sentiment = sentiment.to(device, dtype=torch.long)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            
            fin_outputs.extend(outputs.cpu().detach().numpy())
            fin_targets.extend(sentiment.cpu().detach().numpy())

    return np.array(fin_outputs), np.array(fin_targets)