import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample email data
data = {
    'Email Text': [
        "Congratulations! You've won a $1,000 gift card. Click here to claim.",
        "Your order has been shipped and will arrive soon.",
        "Limited time offer! Get 50% off your purchase today.",
        "Your account has been compromised. Please update your password.",
        "Reminder: Your appointment is scheduled for tomorrow.",
        "Don't miss out on our holiday sale! Up to 70% off.",
        "You've been selected for a special promotion!",
        "Your subscription has been confirmed.",
        "Here's your daily newsletter.",
        "Important: Please verify your account information."
    ],
    'Email Type': [
        "spam",
        "ham",
        "spam",
        "spam",
        "ham",
        "spam",
        "spam",
        "ham",
        "ham",
        "ham"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and labels
X = df['Email Text']
y = df['Email Type']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a TfidfVectorizer and MultinomialNB pipeline
model_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the model pipeline
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model_pipeline, model_file)

# Evaluate the model
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

print("Model saved to 'model.pkl'")
