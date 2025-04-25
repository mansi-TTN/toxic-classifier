from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from toxic_classifier import ToxicClassifier
import os
import json
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
classifier = ToxicClassifier()

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configuration
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded'}), 429

@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze_text():
    try:
        # Handle both JSON and file uploads
        if request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Only CSV files are supported'}), 400
            
            # Save the file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)
            
            try:
                # Process the file
                results = []
                df = pd.read_csv(filepath)
                
                # Validate CSV structure
                if 'text' not in df.columns:
                    return jsonify({'error': 'CSV must contain a "text" column'}), 400
                
                for text in df['text']:
                    if not isinstance(text, str) or not text.strip():
                        continue
                    result = process_single_text(text)
                    results.append(result)
                
                return jsonify({'results': results})
                
            finally:
                # Clean up
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
        else:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            text = data.get('text')
            if not text or not isinstance(text, str) or not text.strip():
                return jsonify({'error': 'Invalid text provided'}), 400
            
            result = process_single_text(text)
            return jsonify(result)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_single_text(text):
    """Process a single text and return comprehensive analysis"""
    try:
        # Get toxicity classification
        toxicity_level, probabilities = classifier.classify_toxicity(text)
        
        # Generate explanations
        explanations = classifier.explain_prediction(text)
        
        # Generate visualization
        plot_data = classifier.visualize_explanation(text)
        
        # Generate counterfactual if text is toxic
        counterfactual_result = None
        if toxicity_level in ['moderate', 'high']:
            counterfactual_result = classifier.generate_counterfactual_with_explanation(text)
        
        # Compile comprehensive response
        result = {
            'text': text,
            'toxicity_analysis': {
                'level': toxicity_level,
                'probabilities': probabilities
            },
            'explanations': explanations,
            'visualization': plot_data,
            'counterfactual': counterfactual_result
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 