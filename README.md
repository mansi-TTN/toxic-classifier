# Toxic Comment Classifier with Counterfactual Generation

This project implements a toxic comment classifier using BERT that can:
1. Classify toxic comments into three levels: high, moderate, and low
2. Generate counterfactuals for toxic comments
3. Use the BERT model for state-of-the-art natural language understanding

## Features

1. **Toxic Comment Classification**
   - Classifies comments into three toxicity levels
   - Provides probability scores for each toxicity level
   - Uses BERT's contextual understanding for better classification

2. **Counterfactual Generation**
   - Generates less toxic alternatives for toxic comments
   - Maintains the original meaning where possible
   - Uses an extensive dictionary of toxic words and alternatives

3. **Model Management**
   - Automatic checkpoint saving
   - Early stopping to prevent overfitting
   - Loads existing model if available
   - Class weights for handling imbalanced data

## Setup

1. **Install Python Dependencies**:
   ```bash
   pip install torch transformers pandas numpy scikit-learn nltk tqdm
   ```

2. **Download NLTK Data**:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

3. **Dataset Setup**:
   - Download the dataset from [Kaggle's Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
   - Place the `train.csv` file in the project directory

## Usage

1. **First Run (Training)**:
   ```bash
   python main.py
   ```
   - This will:
     - Create a 'checkpoints' directory
     - Train the model on a subset of 1000 samples
     - Save checkpoints during training
     - Save the best model
     - Test on example texts

2. **Subsequent Runs**:
   ```bash
   python main.py
   ```
   - This will:
     - Load the existing best model
     - Skip training
     - Test on example texts

## Project Structure

- `main.py`: Main script for training and testing
- `toxic_classifier.py`: Core classifier implementation
- `checkpoints/`: Directory for saved models
- `train.csv`: Dataset file

## Model Details

- Uses BERT (bert-base-uncased) for text classification
- Three toxicity levels: high, moderate, and low
- Batch size: 8 (optimized for faster training)
- Early stopping patience: 3 epochs
- Sample size: 1000 (balanced between toxic and non-toxic)

## Example Output

```
Text: "You are an idiot and I hate you!"
Toxicity level: high
Probabilities: [0.1, 0.2, 0.7]
Counterfactual: "You are a person and I disagree with you!"
Counterfactual toxicity level: low
```

## Notes

- The model requires a GPU for faster training, but will run on CPU if GPU is not available
- Training time on CPU: ~30-60 minutes
- Training time on GPU: ~10-15 minutes
- The counterfactual generation can be expanded by adding more toxic words and their alternatives to the dictionary

## Troubleshooting

1. **Out of Memory Error**:
   - The code automatically handles out-of-memory errors
   - Clears cache and continues training

2. **Model Not Training**:
   - Delete the 'checkpoints' directory to force retraining
   - Check if train.csv is in the correct location

3. **Slow Performance**:
   - Reduce the sample_size parameter in main.py
   - Reduce the batch_size parameter
   - Use a GPU if available

## Contributing

Feel free to:
- Add more toxic words and alternatives to the dictionary
- Modify the toxicity level classification rules
- Improve the counterfactual generation algorithm
- Add more features or optimizations 