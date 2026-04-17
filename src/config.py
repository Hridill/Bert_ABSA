import torch
from transformers import BertTokenizer

# Model Configuration
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "Input/healthcare_sentiment_dataset.csv"
TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Aspect classifier
ASPECT_MODEL_PATH = "model_versions/aspect_classifier.bin"