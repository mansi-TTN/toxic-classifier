import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ToxicClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        self.label_encoder = LabelEncoder()
        
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def classify_toxicity(self, text):
        # Preprocess the text
        text = self.preprocess_text(text)
        
        # Tokenize and prepare input
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move input to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1)

        # Map prediction to toxicity level
        toxicity_levels = ['low', 'moderate', 'high']
        return toxicity_levels[predicted_class.item()], predictions[0].tolist()

    def generate_counterfactual(self, text):
        # Tokenize the text
        words = word_tokenize(text)
        
        # Dictionary of toxic words and their alternatives
        toxic_alternatives = {
            'hate': ['dislike', 'disagree with'],
            'stupid': ['uninformed', 'mistaken'],
            'idiot': ['person', 'individual'],
            'dumb': ['unclear', 'confusing'],
            'kill': ['defeat', 'overcome'],
            # Add more toxic words and their alternatives
        }

        # Generate counterfactual by replacing toxic words
        counterfactual = []
        for word in words:
            word_lower = word.lower()
            if word_lower in toxic_alternatives:
                # Replace with first alternative
                counterfactual.append(toxic_alternatives[word_lower][0])
            else:
                counterfactual.append(word)

        return ' '.join(counterfactual)

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None,
              batch_size=16, epochs=3, learning_rate=2e-5):
        
        # Encode labels
        train_labels_encoded = self.label_encoder.fit_transform(train_labels)
        if val_texts is not None:
            val_labels_encoded = self.label_encoder.transform(val_labels)

        # Create datasets
        train_dataset = ToxicDataset(
            train_texts, train_labels_encoded, self.tokenizer
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        if val_texts is not None:
            val_dataset = ToxicDataset(
                val_texts, val_labels_encoded, self.tokenizer
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )

        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f'Average training loss: {avg_train_loss}')

            # Validation
            if val_texts is not None:
                self.model.eval()
                val_loss = 0
                correct_predictions = 0
                total_predictions = 0

                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)

                        outputs = self.model(
                            input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        val_loss += outputs.loss.item()
                        predictions = torch.argmax(outputs.logits, dim=1)
                        correct_predictions += (predictions == labels).sum().item()
                        total_predictions += labels.shape[0]

                avg_val_loss = val_loss / len(val_loader)
                accuracy = correct_predictions / total_predictions
                print(f'Validation Loss: {avg_val_loss}')
                print(f'Validation Accuracy: {accuracy}') 