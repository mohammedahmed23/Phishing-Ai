import joblib
import pandas as pd

# Load the trained model and vectorizer
model = joblib.load('phishing_email_detector.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_email(email_text):
    # Transform the email text to match the model's input format
    email_text_tfidf = vectorizer.transform([email_text])
    
    # Predict using the trained model
    prediction = model.predict(email_text_tfidf)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    # Replace this with the text of the email you want to check
    email_text = """
    Dear zoya. how are you
    """
    
    result = predict_email(email_text)
    print(f"The email is classified as: {result}")
