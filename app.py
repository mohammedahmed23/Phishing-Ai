from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Define the directory where your pkl files are located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# Load the trained model pipeline
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model_pipeline = pickle.load(model_file)
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

@app.route('/')
def home():
    return "Welcome to the Email Prediction API. Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        email_text = data.get('email_text')
        
        if email_text:
            try:
                # Transform and predict using the loaded model pipeline
                prediction = model_pipeline.predict([email_text])[0]
                
                # Return different responses based on the prediction
                if prediction == 'spam':
                    return jsonify({"prediction": "spam", "message": "This email is classified as spam."})
                elif prediction == 'ham':
                    return jsonify({"prediction": "ham", "message": "This email is classified as non-spam (ham)."})
                else:
                    return jsonify({"error": "Unexpected prediction result."}), 500
            except Exception as e:
                return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        else:
            return jsonify({"error": "Missing 'email_text' field"}), 400
    else:
        return jsonify({"error": "Content-Type must be application/json"}), 415

if __name__ == '__main__':
    app.run(debug=True)
