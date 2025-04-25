import pandas as pd
from sklearn.model_selection import train_test_split
from toxic_classifier import ToxicClassifier

def load_and_prepare_data(file_path):
    # Load the toxic comments dataset
    # Note: You'll need to download the dataset from Kaggle and provide the correct path
    df = pd.read_csv(file_path)
    
    # For this example, we'll use a simplified version with three toxicity levels
    def get_toxicity_level(row):
        if row['toxic'] == 1:
            return 'high'
        elif row['obscene'] == 1 or row['insult'] == 1:
            return 'moderate'
        else:
            return 'low'
    
    df['toxicity_level'] = df.apply(get_toxicity_level, axis=1)
    
    return df['comment_text'].values, df['toxicity_level'].values

def main():
    # Initialize the classifier
    classifier = ToxicClassifier()
    
    try:
        # Load and prepare the data
        # Replace 'path_to_dataset.csv' with your actual dataset path
        texts, labels = load_and_prepare_data('./train.csv')
        
        # Split the data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Train the model
        print("Training the model...")
        classifier.train(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            batch_size=16,
            epochs=3
        )
        
        # Example usage
        test_texts = [
            "You are an idiot and I hate you!",
            "I disagree with your opinion.",
            "This is a neutral comment.",
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