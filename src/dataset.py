import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import numpy as np

# Download all required NLTK data
required_nltk_data = ['punkt', 'wordnet', 'averaged_perceptron_tagger']
for resource in required_nltk_data:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

class BERTDataset:
    def __init__(self, review, target, augment=False):
        self.review = review
        self.target = target
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH)
        self.max_length = config.MAX_LEN
        self.augment = augment

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        target = self.target[item]

        if self.augment and random.random() < 0.3:  # 30% chance of augmentation
            try:
                review = self._augment_text(review)
            except Exception as e:
                # If augmentation fails, use original text
                pass

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'ids': encoding['input_ids'].flatten(),
            'mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'sentiment': torch.tensor(target, dtype=torch.long)
        }

    def _augment_text(self, text):
        """Apply text augmentation techniques."""
        if not text.strip():
            return text

        augmentation_type = random.choice(['synonym_replacement', 'random_deletion', 'random_swap'])
        
        try:
            if augmentation_type == 'synonym_replacement':
                return self._synonym_replacement(text)
            elif augmentation_type == 'random_deletion':
                return self._random_deletion(text)
            else:
                return self._random_swap(text)
        except Exception as e:
            # If any augmentation fails, return original text
            return text

    def _synonym_replacement(self, text, n=1):
        """Replace n words with their synonyms."""
        words = word_tokenize(text)
        if len(words) <= 1:
            return text

        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalnum()]))
        random.shuffle(random_word_list)
        num_replaced = 0
        
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break

        return ' '.join(new_words)

    def _get_synonyms(self, word):
        """Get synonyms for a word."""
        synonyms = []
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word:
                        synonyms.append(synonym)
        except Exception:
            pass
        return list(set(synonyms))

    def _random_deletion(self, text, p=0.1):
        """Randomly delete words with probability p."""
        words = word_tokenize(text)
        if len(words) <= 1:
            return text

        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)

        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return words[rand_int]

        return ' '.join(new_words)

    def _random_swap(self, text, n=1):
        """Randomly swap n pairs of words."""
        words = word_tokenize(text)
        if len(words) <= 1:
            return text

        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) <= 1:
                return ' '.join(new_words)
                
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
        return ' '.join(new_words)