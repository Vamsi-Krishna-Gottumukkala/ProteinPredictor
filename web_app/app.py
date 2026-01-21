from flask import Flask, render_template, request, jsonify
import sys
import os

# Add parent directory to path so we can import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import ProteinPredictionPipeline

app = Flask(__name__)

# Initialize pipeline ONCE when app starts
print("Loading Protein Prediction Pipeline...")
pipeline = ProteinPredictionPipeline()
print("Pipeline Ready!")

@app.route('/')
def home():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API Endpoint for protein structure prediction."""
    try:
        data = request.json
        sequence = data.get('sequence', '')
        
        if not sequence:
            return jsonify({'error': 'No sequence provided'}), 400
            
        if len(sequence) < 10:
             return jsonify({'error': 'Sequence too short (min 10 residues)'}), 400
             
        # Run prediction
        result = pipeline.predict(sequence)
        
        return jsonify({
            'success': True,
            'pdb': result['pdb_content'],
            'metrics': result['metrics'],
            'ss': result['secondary_structure']
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
