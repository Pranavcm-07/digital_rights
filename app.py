from flask import Flask, request, jsonify
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load dataset
dataset = pd.read_csv("dataset_cleaned.csv")

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens) if tokens else 'empty'

# Apply preprocessing to dataset
dataset['summary'] = dataset['summary'].fillna('')
dataset['clean_summary'] = dataset['summary'].apply(preprocess_text)

# Prepare features and labels
X = dataset['clean_summary']
y = dataset['condition']  # 0 = Non-compliant, 1 = Compliant

# Convert text to numerical vectors
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Train various models
# -------------------------------

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f'Random Forest Accuracy: {rf_accuracy:.2f}')
print(classification_report(y_test, rf_pred))

# XGBoost
print("XGBoost")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f'XGBoost Accuracy: {xgb_accuracy:.2f}')
print(classification_report(y_test, xgb_pred))

# Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f'Logistic Regression Accuracy: {lr_accuracy:.2f}')
print(classification_report(y_test, lr_pred))

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
print(f'Naive Bayes Accuracy: {nb_accuracy:.2f}')
print(classification_report(y_test, nb_pred))

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f'Decision Tree Accuracy: {dt_accuracy:.2f}')
print(classification_report(y_test, dt_pred))

# -------------------------------
# Select the best model based on accuracy
# -------------------------------
model_accuracies = {
    'Random Forest': rf_accuracy,
    'XGBoost': xgb_accuracy,
    'Logistic Regression': lr_accuracy,
    'Naive Bayes': nb_accuracy,
    'Decision Tree': dt_accuracy
}

models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'Logistic Regression': lr_model,
    'Naive Bayes': nb_model,
    'Decision Tree': dt_model
}

best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = models[best_model_name]
print(f'Best model selected: {best_model_name} with accuracy {model_accuracies[best_model_name]:.2f}')

# -------------------------------
# Define prediction endpoint
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid request'}), 400

    # Preprocess input text
    text = preprocess_text(data['text'])
    if text == 'empty':
        return jsonify({'error': 'Input text is too short or empty after preprocessing'}), 400

    print(f"Preprocessed Input Text: {text}")  # Debugging line
    vectorized_text = vectorizer.transform([text])

    # Get prediction from the best model
    prediction = best_model.predict(vectorized_text)[0]

    # Map prediction to a label
    label_map = {0: "Non-compliant", 1: "Compliant"}
    prediction_label = label_map[prediction]

    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)
