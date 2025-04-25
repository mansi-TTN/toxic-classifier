# Toxic Comment Classifier with Counterfactual Generation

This project implements a toxic comment classifier using BERT that can:
1. Classify toxic comments into three levels: high, moderate, and low
2. Generate counterfactuals for toxic comments
3. Use the BERT model for state-of-the-art natural language understanding

## Setup

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Go to [Kaggle's Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
   - Download the dataset
   - Place the train.csv file in the project directory

## Usage

1. Update the dataset path in `main.py`:
```python
texts, labels = load_and_prepare_data('path_to_dataset.csv')
```

2. Run the main script:
```bash
python main.py
```

The script will:
- Train the model on the toxic comments dataset
- Demonstrate classification on example texts
- Generate and evaluate counterfactuals for toxic comments

## Model Details

The classifier uses:
- BERT (bert-base-uncased) for text classification
- Three toxicity levels: high, moderate, and low
- Counterfactual generation through word replacement
- PyTorch for deep learning implementation

## Features

1. **Toxic Comment Classification**
   - Classifies comments into three toxicity levels
   - Provides probability scores for each toxicity level
   - Uses BERT's contextual understanding for better classification

2. **Counterfactual Generation**
   - Generates less toxic alternatives for toxic comments
   - Maintains the original meaning where possible
   - Validates the generated counterfactuals

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
- Initial training might take several hours depending on the dataset size and hardware
- The counterfactual generation can be expanded by adding more toxic words and their alternatives 