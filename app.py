from flask import Flask, request, jsonify
from toxic_classifier import ToxicClassifier
import os

app = Flask(__name__)

# Initialize the classifier
classifier = ToxicClassifier()

# Check if model exists, if not train it
best_model_path = os.path.join('checkpoints', 'best_model.pt')
if not os.path.exists(best_model_path):
    print("No existing model found. Please train the model first using main.py")
    exit(1)

# Load the best model
classifier.load_checkpoint(best_model_path)

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        toxicity_level, probabilities = classifier.classify_toxicity(text)
        
        return jsonify({
            'text': text,
            'toxicity_level': toxicity_level,
            'probabilities': probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_counterfactual', methods=['POST'])
def generate_counterfactual():
    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        counterfactual = classifier.generate_counterfactual(text)
        counter_level, counter_probs = classifier.classify_toxicity(counterfactual)
        
        return jsonify({
            'original_text': text,
            'counterfactual': counterfactual,
            'counterfactual_toxicity_level': counter_level,
            'counterfactual_probabilities': counter_probs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    try:
        data = request.get_json()
        if 'texts' not in data or not isinstance(data['texts'], list):
            return jsonify({'error': 'No texts array provided'}), 400
        
        results = []
        for text in data['texts']:
            toxicity_level, probabilities = classifier.classify_toxicity(text)
            results.append({
                'text': text,
                'toxicity_level': toxicity_level,
                'probabilities': probabilities
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 