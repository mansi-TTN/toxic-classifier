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
import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

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
        
        # Initialize XAI framework
        self.lime_explainer = LimeTextExplainer(class_names=['low', 'moderate', 'high'])
        
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize counterfactual generation model
        self.counterfactual_model = None
        self.toxic_patterns = {}
        self.non_toxic_patterns = {}
        
        # Semantic word mappings for context preservation
        self.semantic_mappings = {
            # Personal attacks
            'idiot': 'person',
            'stupid': 'incorrect',
            'moron': 'individual',
            'dumb': 'mistaken',
            'fool': 'someone',
            
            # Strong negative emotions
            'hate': 'disagree',
            'loathe': 'dislike',
            'despise': 'disapprove',
            
            # Insults
            'jerk': 'person',
            'asshole': 'individual',
            'bastard': 'person',
            
            # Aggressive terms
            'kill': 'stop',
            'destroy': 'change',
            'ruin': 'affect',
            
            # General negative terms
            'terrible': 'different',
            'awful': 'unusual',
            'horrible': 'unexpected'
        }
        
        # Word categories for better context matching
        self.word_categories = {
            'noun': ['person', 'individual', 'someone', 'one'],
            'verb': ['disagree', 'dislike', 'disapprove', 'differ'],
            'adjective': ['incorrect', 'mistaken', 'different', 'unusual']
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
        """Preprocess text for classification"""
        try:
            # Convert to lowercase
            text = text.lower()
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            return ""

    def classify_toxicity(self, text):
        """Classify text toxicity"""
        try:
            # Validate input
            if not text or not isinstance(text, str) or len(text.strip()) < 3:
                return 'error', [0.33, 0.33, 0.33]
            
            # Preprocess the text
            text = self.preprocess_text(text)
            
            # Skip if text is too short after preprocessing
            if len(text.strip()) < 3:
                return 'error', [0.33, 0.33, 0.33]
            
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
            probabilities = [float(p) for p in predictions[0].tolist()]
            level = toxicity_levels[predicted_class.item()]
            
            return level, probabilities
            
        except Exception as e:
            print(f"Error in classification: {str(e)}")
            return 'error', [0.33, 0.33, 0.33]

    def learn_counterfactual_patterns(self, texts, labels):
        """Learn patterns from the dataset for counterfactual generation"""
        from collections import defaultdict
        import re
        
        # Initialize pattern dictionaries
        toxic_words = defaultdict(list)
        non_toxic_words = defaultdict(list)
        
        # Process each text with progress bar
        print("Learning patterns from dataset...")
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Learning patterns"):
            # Tokenize and clean text
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            
            # Store patterns based on toxicity
            if label == 'high' or label == 'moderate':
                for word in words:
                    toxic_words[word].extend(words)  # Store all words from toxic texts
            else:
                for word in words:
                    non_toxic_words[word].extend(words)  # Store all words from non-toxic texts
        
        # Store learned patterns
        self.toxic_patterns = dict(toxic_words)
        self.non_toxic_patterns = dict(non_toxic_words)
        
        # Print statistics about learned patterns
        print(f"Learned {len(self.toxic_patterns)} toxic patterns")
        print(f"Learned {len(self.non_toxic_patterns)} non-toxic patterns")

    def get_word_category(self, word):
        """Get the grammatical category of a word"""
        pos = nltk.pos_tag([word])[0][1]
        if pos.startswith('NN'):  # Noun
            return 'noun'
        elif pos.startswith('VB'):  # Verb
            return 'verb'
        elif pos.startswith('JJ'):  # Adjective
            return 'adjective'
        return None

    def generate_counterfactual(self, text):
        """Generate counterfactual text"""
        try:
            # Enhanced semantic mappings for better replacements
            semantic_mappings = {
                # Profanities and offensive words
                'fuck': 'forget', 'fucking': 'very', 'fucked': 'messed up',
                'shit': 'stuff', 'damn': 'darn', 'hell': 'heck',
                'ass': 'person', 'butt': 'person', 'crap': 'stuff',
                'piss': 'upset', 'pissed': 'upset', 'bitch': 'person',
                'dick': 'person', 'cock': 'person', 'pussy': 'person',
                'bastard': 'person', 'motherfucker': 'person',
                'screw': 'forget', 'screwed': 'messed up',
                'bullshit': 'nonsense', 'crap': 'stuff', 'darn': 'darn',
                'damnit': 'darn', 'goddamn': 'darn', 'bloody': 'very',
                'bugger': 'person', 'sod': 'person', 'twat': 'person',
                'wanker': 'person', 'arse': 'person', 'arsehole': 'person',
                'bellend': 'person', 'knob': 'person', 'knobhead': 'person',
                'prick': 'person', 'tosser': 'person', 'dipshit': 'person',
                'dumbass': 'person', 'jackass': 'person', 'shithead': 'person',
                'shitface': 'person', 'shitbag': 'person', 'shitstain': 'person',
                'fuckface': 'person', 'fuckhead': 'person', 'fuckwit': 'person',
                'fucknut': 'person', 'fucktard': 'person', 'fuckup': 'person',
                'fuckwad': 'person', 'fuckstick': 'person', 'fuckbucket': 'person',
                'fucknugget': 'person', 'fuckbrain': 'person', 'fuckhole': 'person',
                'fuckbag': 'person', 'fucktoy': 'person', 'fuckboy': 'person',
                'fuckgirl': 'person', 'fucktard': 'person', 'fuckup': 'person',
                'fuckwad': 'person', 'fuckstick': 'person', 'fuckbucket': 'person',
                'fucknugget': 'person', 'fuckbrain': 'person', 'fuckhole': 'person',
                'fuckbag': 'person', 'fucktoy': 'person', 'fuckboy': 'person',
                'fuckgirl': 'person',
                
                # Personal attacks
                'idiot': 'person', 'stupid': 'incorrect', 'moron': 'individual',
                'dumb': 'mistaken', 'fool': 'someone', 'jerk': 'person',
                'asshole': 'individual', 'retard': 'person', 'cunt': 'person',
                'imbecile': 'person', 'dunce': 'person', 'nincompoop': 'person',
                'blockhead': 'person', 'bonehead': 'person', 'dullard': 'person',
                'simpleton': 'person', 'halfwit': 'person', 'dimwit': 'person',
                'nitwit': 'person', 'numbskull': 'person', 'dumbass': 'person',
                'dumbfuck': 'person', 'dumbhead': 'person', 'dumbnut': 'person',
                'dumbshit': 'person', 'dumbstruck': 'person', 'dumbwaiter': 'person',
                'dumbfound': 'person', 'dumbfounder': 'person', 'dumbfoundest': 'person',
                
                # Strong negative emotions
                'hate': 'dislike', 'loathe': 'disapprove', 'despise': 'disagree',
                'abhor': 'disapprove', 'detest': 'dislike', 'abominate': 'dislike',
                'execrate': 'dislike', 'revile': 'dislike', 'vilify': 'dislike',
                'denounce': 'disapprove', 'condemn': 'disapprove', 'censure': 'disapprove',
                'reproach': 'disapprove', 'rebuke': 'disapprove', 'reprimand': 'disapprove',
                'reprove': 'disapprove', 'upbraid': 'disapprove', 'berate': 'disapprove',
                'castigate': 'disapprove', 'chastise': 'disapprove', 'scold': 'disapprove',
                
                # Aggressive terms
                'kill': 'stop', 'destroy': 'change', 'ruin': 'affect',
                'murder': 'stop', 'attack': 'address', 'assault': 'confront',
                'annihilate': 'stop', 'eliminate': 'remove', 'eradicate': 'remove',
                'exterminate': 'remove', 'extinguish': 'stop', 'obliterate': 'remove',
                'slay': 'stop', 'slaughter': 'stop', 'massacre': 'stop',
                'butcher': 'stop', 'execute': 'stop', 'dispatch': 'stop',
                
                # Insults and slurs
                'nigger': 'person', 'nigga': 'person', 'fag': 'person',
                'faggot': 'person', 'whore': 'person', 'slut': 'person',
                'spic': 'person', 'kike': 'person', 'chink': 'person',
                'gook': 'person', 'wetback': 'person', 'beaner': 'person',
                'towelhead': 'person', 'sandnigger': 'person', 'cameljockey': 'person',
                'raghead': 'person', 'taco': 'person', 'cholo': 'person',
                'spook': 'person', 'coon': 'person', 'jigaboo': 'person',
                'junglebunny': 'person', 'porchmonkey': 'person', 'spearchucker': 'person',
                'zipperhead': 'person', 'gook': 'person', 'slant': 'person',
                'yellow': 'person', 'chink': 'person', 'gook': 'person',
                
                # General negative terms
                'terrible': 'different', 'awful': 'unusual', 'horrible': 'unexpected',
                'disgusting': 'unpleasant', 'vile': 'unpleasant', 'nasty': 'unpleasant',
                'revolting': 'unpleasant', 'repulsive': 'unpleasant', 'repugnant': 'unpleasant',
                'abhorrent': 'unpleasant', 'detestable': 'unpleasant', 'loathsome': 'unpleasant',
                'odious': 'unpleasant', 'offensive': 'unpleasant', 'obnoxious': 'unpleasant',
                'noxious': 'unpleasant', 'putrid': 'unpleasant', 'rancid': 'unpleasant',
                'rotten': 'unpleasant', 'stinking': 'unpleasant', 'tainted': 'unpleasant'
            }
            
            # Tokenize the text while preserving punctuation
            words = text.split()
            result = []
            changes = []
            
            for word in words:
                # Convert to lowercase for comparison but preserve original case
                word_lower = word.lower()
                
                # Check if word is in semantic mappings
                if word_lower in semantic_mappings:
                    replacement = semantic_mappings[word_lower]
                    # Preserve original capitalization
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    result.append(replacement)
                    changes.append(f"{word} -> {replacement}")
                else:
                    result.append(word)
            
            # Join words back into text
            counterfactual = ' '.join(result)
            
            # Post-processing to fix common issues
            counterfactual = re.sub(r'\s+([.,!?])', r'\1', counterfactual)  # Fix spacing around punctuation
            counterfactual = re.sub(r'\s+', ' ', counterfactual)  # Fix multiple spaces
            
            return {
                'text': counterfactual,
                'changes': changes
            }
            
        except Exception as e:
            print(f"Error in counterfactual generation: {str(e)}")
            return {
                'text': text,
                'changes': []
            }

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

    def explain_prediction(self, text):
        """Generate explanations using LIME"""
        try:
            # Validate input
            if not text or not isinstance(text, str) or len(text.strip()) < 3:
                return None
            
            # Get base prediction
            toxicity_level, probabilities = self.classify_toxicity(text)
            
            # Generate LIME explanation with reduced perturbations
            def predict_proba(texts):
                results = []
                for t in texts:
                    if not t or len(t.strip()) < 3:  # Skip empty or too short texts
                        results.append([0.33, 0.33, 0.33])
                        continue
                    _, probs = self.classify_toxicity(t)
                    results.append(probs)
                return np.array(results)
            
            lime_exp = self.lime_explainer.explain_instance(
                text,
                predict_proba,
                num_features=5,  # Reduced from 10 to 5
                num_samples=100  # Reduced number of perturbations
            )
            
            # Convert numpy arrays to lists for JSON serialization
            probabilities = [float(p) for p in probabilities]  # Convert numpy float32 to Python float
            local_pred = [float(p) for p in lime_exp.local_pred]  # Convert numpy array to list
            
            # Format explanations
            explanations = {
                'toxicity_level': toxicity_level,
                'probabilities': probabilities,
                'lime_explanation': {
                    'important_features': lime_exp.as_list(),
                    'local_prediction': local_pred
                }
            }
            
            return explanations
            
        except Exception as e:
            print(f"Error in explanation generation: {str(e)}")
            return None

    def visualize_explanation(self, text):
        """Generate visualization of explanations"""
        try:
            explanations = self.explain_prediction(text)
            if not explanations:
                return None
            
            # Create visualization
            fig = plt.figure(figsize=(15, 10))
            
            try:
                # Plot 1: LIME Feature Importance
                plt.subplot(2, 1, 1)
                lime_features = explanations['lime_explanation']['important_features']
                features = [f[0] for f in lime_features]
                importance = [f[1] for f in lime_features]
                sns.barplot(x=importance, y=features)
                plt.title('Feature Importance (LIME)')
                
                # Plot 2: Probability Distribution
                plt.subplot(2, 1, 2)
                probs = explanations['probabilities']
                labels = ['low', 'moderate', 'high']
                sns.barplot(x=labels, y=probs)
                plt.title('Toxicity Probability Distribution')
                
                plt.tight_layout()
                
                # Save plot to bytes
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                
                return plot_data
                
            finally:
                plt.close(fig)
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            return None

    def generate_counterfactual_with_explanation(self, text):
        """Generate counterfactual with explanation of changes"""
        try:
            # Generate counterfactual
            counterfactual = self.generate_counterfactual(text)
            
            # Get explanations for both original and counterfactual
            original_exp = self.explain_prediction(text)
            counterfactual_exp = self.explain_prediction(counterfactual['text'])
            
            # Compare changes
            changes = []
            original_words = word_tokenize(text.lower())
            counterfactual_words = word_tokenize(counterfactual['text'].lower())
            
            for orig_word, cf_word in zip(original_words, counterfactual_words):
                if orig_word != cf_word:
                    changes.append({
                        'original': orig_word,
                        'replacement': cf_word,
                        'reason': 'Toxic word replaced with non-toxic alternative'
                    })
            
            return {
                'original_text': text,
                'counterfactual': counterfactual['text'],
                'changes': changes,
                'original_explanation': original_exp,
                'counterfactual_explanation': counterfactual_exp
            }
            
        except Exception as e:
            print(f"Error in counterfactual explanation: {str(e)}")
            return None

    def analyze_text(self, text):
        """Analyze text and generate counterfactual if toxic"""
        try:
            # Classify toxicity
            toxicity_level, probabilities = self.classify_toxicity(text)
            
            # Generate counterfactual if toxic
            counterfactual_result = None
            if toxicity_level in ['moderate', 'high']:
                counterfactual_result = self.generate_counterfactual(text)
            
            # Prepare response
            result = {
                'text': text,
                'toxicity_analysis': {
                    'level': toxicity_level,
                    'probabilities': probabilities
                },
                'counterfactual': counterfactual_result,
                'counterfact': counterfactual_result['text'] if counterfactual_result else None
            }
            
            return result
            
        except Exception as e:
            print(f"Error in text analysis: {str(e)}")
            return {
                'error': str(e)
            } 