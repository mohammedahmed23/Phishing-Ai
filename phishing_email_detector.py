import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset and skip the index column
df = pd.read_csv('emails.csv', index_col=0)

# Debugging: Print column names and first few rows
print("Column names:", df.columns.tolist())
print("First few rows:")
print(df.head())

# Handle missing values in 'Email Text'
df['Email Text'] = df['Email Text'].fillna('No content')  # Fill missing values with a placeholder

# Check for any remaining missing values
print("Missing values in 'Email Text':", df['Email Text'].isna().sum())

# Use the correct column names
try:
    X = df['Email Text']  # Column with email text
    y = df['Email Type']  # Column with email labels
except KeyError as e:
    print(f"KeyError: {e} - Check column names in your dataset.")
    raise  # Re-raise the exception for further debugging if needed

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Prediction
y_pred = model.predict(X_test_tfidf)

# Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'phishing_email_detector.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
