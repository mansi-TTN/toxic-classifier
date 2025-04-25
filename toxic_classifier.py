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
import os
from pathlib import Path
import gc

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
    def __init__(self, model_name='bert-base-uncased', num_labels=3, checkpoint_dir='checkpoints'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        self.label_encoder = LabelEncoder()
        self.checkpoint_dir = checkpoint_dir
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

        # Expanded toxic words dictionary
        self.toxic_alternatives = {
            'hate': ['dislike', 'disagree with', 'have different views from'],
            'stupid': ['uninformed', 'mistaken', 'incorrect'],
            'idiot': ['person', 'individual', 'someone'],
            'dumb': ['unclear', 'confusing', 'difficult to understand'],
            'kill': ['defeat', 'overcome', 'surpass'],
            'moron': ['person', 'individual', 'someone'],
            'retard': ['person', 'individual', 'someone'],
            'fool': ['person', 'individual', 'someone'],
            'worthless': ['needs improvement', 'could be better'],
            'useless': ['needs work', 'requires attention'],
            'pathetic': ['unfortunate', 'disappointing'],
            'disgusting': ['unpleasant', 'not preferred'],
            'horrible': ['not good', 'needs improvement'],
            'terrible': ['poor', 'needs work'],
            'awful': ['not satisfactory', 'needs attention'],
            'crap': ['poor quality', 'needs improvement'],
            'suck': ['not good', 'needs work'],
            'loser': ['person', 'individual', 'someone'],
            'failure': ['learning experience', 'opportunity for growth'],
            'waste': ['needs improvement', 'could be better'],
            'garbage': ['poor quality', 'needs work']
        }

    def save_checkpoint(self, epoch, optimizer, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pt'))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['optimizer_state_dict']

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def classify_toxicity(self, text):
        try:
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
        except Exception as e:
            print(f"Error in classification: {str(e)}")
            return 'error', [0.33, 0.33, 0.33]

    def generate_counterfactual(self, text):
        try:
            # Tokenize the text
            words = word_tokenize(text)
            
            # Generate counterfactual by replacing toxic words
            counterfactual = []
            for word in words:
                word_lower = word.lower()
                if word_lower in self.toxic_alternatives:
                    # Replace with random alternative
                    counterfactual.append(np.random.choice(self.toxic_alternatives[word_lower]))
                else:
                    counterfactual.append(word)

            return ' '.join(counterfactual)
        except Exception as e:
            print(f"Error in counterfactual generation: {str(e)}")
            return text

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None,
              batch_size=16, epochs=3, learning_rate=2e-5, patience=3):
        
        try:
            # Encode labels
            train_labels_encoded = self.label_encoder.fit_transform(train_labels)
            if val_texts is not None:
                val_labels_encoded = self.label_encoder.transform(val_labels)

            # Calculate class weights for imbalanced data
            class_counts = np.bincount(train_labels_encoded)
            class_weights = 1. / class_counts
            class_weights = torch.FloatTensor(class_weights).to(self.device)

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

            # Early stopping variables
            best_val_loss = float('inf')
            patience_counter = 0
            best_epoch = 0

            # Training loop
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                
                # Progress bar for training
                train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
                
                for batch in train_pbar:
                    try:
                        optimizer.zero_grad()
                        
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)

                        outputs = self.model(
                            input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        # Apply class weights
                        loss = outputs.loss * class_weights[labels].mean()
                        total_loss += loss.item()
                        
                        loss.backward()
                        optimizer.step()

                        # Update progress bar
                        train_pbar.set_postfix({'loss': loss.item()})

                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print('Out of memory error. Clearing cache and continuing...')
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        else:
                            raise e

                avg_train_loss = total_loss / len(train_loader)
                print(f'Average training loss: {avg_train_loss}')

                # Validation
                if val_texts is not None:
                    self.model.eval()
                    val_loss = 0
                    correct_predictions = 0
                    total_predictions = 0

                    with torch.no_grad():
                        val_pbar = tqdm(val_loader, desc='Validation')
                        for batch in val_pbar:
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

                            # Update progress bar
                            val_pbar.set_postfix({'val_loss': outputs.loss.item()})

                    avg_val_loss = val_loss / len(val_loader)
                    accuracy = correct_predictions / total_predictions
                    print(f'Validation Loss: {avg_val_loss}')
                    print(f'Validation Accuracy: {accuracy}')

                    # Early stopping check
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_epoch = epoch
                        patience_counter = 0
                        self.save_checkpoint(epoch, optimizer, avg_val_loss, is_best=True)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f'Early stopping triggered after {epoch + 1} epochs')
                            print(f'Best model was from epoch {best_epoch + 1}')
                            break

                # Save checkpoint every epoch
                self.save_checkpoint(epoch, optimizer, avg_train_loss)

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise e 