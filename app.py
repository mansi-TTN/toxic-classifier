from flask import Flask, request, jsonify
from toxic_classifier import ToxicClassifier
import pandas as pd
import torch
import gc

app = Flask(__name__)
classifier = ToxicClassifier()

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text']
        result = classifier.analyze_text(text)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400
            
        df = pd.read_csv(file)
        if 'text' not in df.columns:
            return jsonify({'error': 'CSV must contain a "text" column'}), 400
            
        results = []
        for text in df['text']:
            result = classifier.analyze_text(text)
            results.append(result)
            
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 