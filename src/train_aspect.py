import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

import config
from aspect_model import AspectClassifier
from aspect_utils import ASPECT_LABELS, heuristic_detect_aspect
from logger import setup_logger


logger = setup_logger('aspect_training')


class AspectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name='distilbert-base-uncased', max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def build_weak_aspect_labels(df):
    # If dataset already has an 'aspect' column, use it; else generate from heuristic
    if 'aspect' in df.columns:
        labels = df['aspect'].fillna('overall').map({a: i for i, a in enumerate(ASPECT_LABELS)}).fillna(-1)
        # filter unknown aspects
        mask = labels != -1
        return df.loc[mask, 'review'], labels[mask]
    else:
        aspects = df['review'].apply(heuristic_detect_aspect)
        mask = aspects != 'overall'
        mapped = aspects[mask].map({a: i for i, a in enumerate(ASPECT_LABELS)})
        return df.loc[mask, 'review'], mapped


def run():
    logger.info('Loading dataset for aspect training')
    df = pd.read_csv(config.TRAINING_FILE).fillna('')
    texts, labels = build_weak_aspect_labels(df)
    logger.info(f'Training examples: {len(texts)}')

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42, stratify=labels)

    train_ds = AspectDataset(X_train, y_train, max_len=config.MAX_LEN)
    val_ds = AspectDataset(X_val, y_val, max_len=config.MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.VALID_BATCH_SIZE)

    device = config.DEVICE
    model = AspectClassifier(num_labels=len(ASPECT_LABELS)).to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    num_train_steps = len(train_loader) * max(1, config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_train_steps), num_training_steps=num_train_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(config.EPOCHS):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        logger.info(f'Epoch {epoch+1} train loss: {running/len(train_loader):.4f}')

        # Eval
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                logits = model(input_ids, attention_mask)
                pred = torch.argmax(logits, dim=1)
                preds.extend(pred.cpu().tolist())
                gts.extend(labels.cpu().tolist())
        report = classification_report(gts, preds, target_names=ASPECT_LABELS, zero_division=0, output_dict=False)
        logger.info('\n' + report)

    # Save
    os.makedirs(os.path.dirname(config.ASPECT_MODEL_PATH), exist_ok=True) if os.path.dirname(config.ASPECT_MODEL_PATH) else None
    torch.save(model.state_dict(), config.ASPECT_MODEL_PATH)
    logger.info(f'Saved aspect model to {config.ASPECT_MODEL_PATH}')


if __name__ == '__main__':
    run()

