import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
df = pd.read_csv("/content/spam.csv", encoding="latin-1", usecols=["v1", "v2"])
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Preprocess data
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].str.replace('[^\w\s]', '').str.lower()

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Feature Extraction
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# SVM with TF-IDF
svm_model_tfidf = SVC(kernel='linear')
svm_model_tfidf.fit(X_train_tfidf, y_train)
svm_predictions_tfidf = svm_model_tfidf.predict(X_test_tfidf)

# Evaluate SVM with TF-IDF
accuracy = accuracy_score(y_test, svm_predictions_tfidf)
precision = precision_score(y_test, svm_predictions_tfidf)
recall = recall_score(y_test, svm_predictions_tfidf)
f1 = f1_score(y_test, svm_predictions_tfidf)

print(f"SVM with TF-IDF: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1-Score={f1}")

# Function to classify a new message
def classify_message(message):
    # Preprocess the message
    message = message.replace('[^\w\s]', '').lower()
    
    # TF-IDF feature
    message_tfidf = tfidf_vectorizer.transform([message])
    
    # Predict using SVM with TF-IDF
    svm_prediction_tfidf = svm_model_tfidf.predict(message_tfidf)
    
    return 'spam' if svm_prediction_tfidf[0] else 'ham'

# Example usage
message = input("Enter a message to classify: ")
prediction = classify_message(message)
print(f"The message is classified as: {prediction}")
