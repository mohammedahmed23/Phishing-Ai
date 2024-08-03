import joblib

def check_email_safety(email_content):
    """
    Checks if the provided email content is safe or phishing using the trained model.
    
    Args:
    email_content (str): The content of the email to be checked.

    Returns:
    str: The prediction result ('Safe Email' or 'Phishing Email').
    """
    # Load the saved model and vectorizer
    model = joblib.load('phishing_email_detector.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Transform the email content using the vectorizer
    email_tfidf = vectorizer.transform([email_content])

    # Predict the type of the email
    prediction = model.predict(email_tfidf)

    # Output the prediction
    return prediction[0]

# Example usage
if __name__ == "__main__":
    # Sample email content
    email_content = "Congratulations! You've won a prize. Click here to claim it."

    # Check email safety
    result = check_email_safety(email_content)
    print("Prediction:", result)
