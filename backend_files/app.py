from flask import Flask, request, jsonify
import joblib
import pandas as pd
import io

app = Flask(__name__)

# Load the serialized pipeline model from the local backend folder directory context
model = joblib.load('superkart_model.joblib')

@app.route('/v1/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input payload
        data = request.get_json(force=True)
        # Convert payload to DataFrame format
        df = pd.DataFrame([data])
        # Generate prediction
        prediction = model.predict(df)[0]
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/v1/predictbatch', methods=['POST'])
def predict_batch():
    try:
        # Check if file is part of the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        # Read CSV file directly from stream
        stream = io.StringIO(file.stream.read().decode("utf-8"), newline=None)
        df = pd.read_csv(stream)

        # Generate predictions
        predictions = model.predict(df)

        # Format predictions as a dictionary mapping index to predicted value
        result = {str(i): float(pred) for i, pred in enumerate(predictions)}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)