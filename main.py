import pandas as pd
from sklearn.model_selection import train_test_split
from toxic_classifier import ToxicClassifier
import os

def load_and_prepare_data(file_path, sample_size=1000):
    # Load the toxic comments dataset
    df = pd.read_csv(file_path)
    
    # Take a smaller sample for faster training
    # Ensure we have a good distribution of toxic and non-toxic comments
    toxic_samples = df[df['toxic'] == 1].sample(min(sample_size//2, len(df[df['toxic'] == 1])))
    non_toxic_samples = df[df['toxic'] == 0].sample(min(sample_size//2, len(df[df['toxic'] == 0])))
    df_sample = pd.concat([toxic_samples, non_toxic_samples])
    
    # For this example, we'll use a simplified version with three toxicity levels
    def get_toxicity_level(row):
        if row['toxic'] == 1:
            return 'high'
        elif row['obscene'] == 1 or row['insult'] == 1:
            return 'moderate'
        else:
            return 'low'
    
    df_sample['toxicity_level'] = df_sample.apply(get_toxicity_level, axis=1)
    
    print(f"Using {len(df_sample)} samples for training")
    print(f"Toxicity distribution:")
    print(df_sample['toxicity_level'].value_counts())
    
    return df_sample['comment_text'].values, df_sample['toxicity_level'].values

def main():
    # Initialize the classifier
    classifier = ToxicClassifier()
    
    try:
        # Check if best model exists
        best_model_path = os.path.join('checkpoints', 'best_model.pt')
        if os.path.exists(best_model_path):
            print("Loading existing best model...")
            classifier.load_checkpoint(best_model_path)
        else:
            print("No existing model found. Starting training...")
            # Load and prepare the data with smaller sample size
            texts, labels = load_and_prepare_data('./train.csv', sample_size=1000)  # Using 1000 samples
            
            # Split the data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            # Train the model with smaller batch size for faster training
            print("Training the model...")
            classifier.train(
                train_texts,
                train_labels,
                val_texts,
                val_labels,
                batch_size=8,  # Reduced batch size
                epochs=3
            )
        
        # Example usage
        test_texts = [
            "You are an idiot and I hate you!",
            "I disagree with your opinion.",
            "This is a neutral comment.",
            "You're so stupid, I can't believe you said that!",
            "That's a terrible idea, you moron!",
        ]
        
        print("\nTesting the classifier with example texts:")
        for text in test_texts:
            toxicity_level, probabilities = classifier.classify_toxicity(text)
            print(f"\nText: {text}")
            print(f"Toxicity level: {toxicity_level}")
            print(f"Probabilities: {probabilities}")
            
            # Generate counterfactual
            counterfactual = classifier.generate_counterfactual(text)
            print(f"Counterfactual: {counterfactual}")
            
            # Classify the counterfactual
            counter_level, counter_probs = classifier.classify_toxicity(counterfactual)
            print(f"Counterfactual toxicity level: {counter_level}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 