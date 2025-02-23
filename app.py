# from flask import Flask, request, jsonify
# import pandas as pd
# import re
# import logging
# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend for servers
# import matplotlib.pyplot as plt
# from collections import Counter

# # Scikit-learn & imblearn imports
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import FunctionTransformer
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
# from sklearn.decomposition import TruncatedSVD

# # Classifiers
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC

# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline  # imblearn pipeline to work with SMOTE

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# # --------------------------------------
# # Custom Preprocess Function
# # --------------------------------------
# def preprocess_text(text):
#     """
#     Preprocess a single document: lowercases text and removes non-alphabetic characters.
#     """
#     text = text.lower()
#     # Remove any character that is not a letter or whitespace
#     text = re.sub(r'[^a-z\s]', '', text)
#     return text

# # --------------------------------------
# # Diagnostic Functions
# # --------------------------------------
# def check_duplicates(df):
#     duplicate_count = df.duplicated().sum()
#     logger.debug(f"Total duplicate rows in dataset: {duplicate_count}")
#     print(f"Total duplicate rows in dataset: {duplicate_count}")

# def analyze_token_frequencies(text_series, y):
#     def get_token_counts(text_series):
#         tokens = []
#         for text in text_series:
#             tokens.extend(re.findall(r'\b\w+\b', text.lower()))
#         return Counter(tokens)
    
#     tokens_class0 = get_token_counts(text_series[y == 0])
#     tokens_class1 = get_token_counts(text_series[y == 1])
#     logger.debug("Top 10 tokens for class 0: %s", tokens_class0.most_common(10))
#     logger.debug("Top 10 tokens for class 1: %s", tokens_class1.most_common(10))
#     print("Top 10 tokens for class 0:", tokens_class0.most_common(10))
#     print("Top 10 tokens for class 1:", tokens_class1.most_common(10))

# def visualize_feature_space(X, y):
#     # Reduce dimensions to 2 for visualization
#     svd = TruncatedSVD(n_components=2, random_state=42)
#     X_reduced = svd.fit_transform(X)
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.7)
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.title("TruncatedSVD Projection of TF-IDF Features")
#     plt.colorbar(scatter, label="Class")
#     plt.savefig("feature_space.png")
#     plt.close()
#     logger.debug("Feature space plot saved as 'feature_space.png'")
#     print("Feature space plot saved as 'feature_space.png'")

# # --------------------------------------
# # Load and Prepare Dataset
# # --------------------------------------
# df1 = pd.read_csv("gdpr_violations_noisy.csv")
# df2 = pd.read_csv("gdpr_text_noisy.csv")
# dataset = pd.concat([df1, df2]).drop_duplicates()
# dataset['summary'] = dataset['summary'].fillna('')

# check_duplicates(dataset)
# X_text = dataset['summary']
# y = dataset['condition']  # 0 = Non-compliant, 1 = Compliant

# print("Original class distribution:")
# print(y.value_counts())
# logger.debug("Original class distribution:\n%s", y.value_counts().to_string())

# analyze_token_frequencies(X_text, y)

# # Split into training and test sets (stratified)
# X_train, X_test, y_train, y_test = train_test_split(
#     X_text, y, test_size=0.2, random_state=42, stratify=y
# )

# # Visualize feature space using TF-IDF on training data (apply custom preprocessor)
# tfidf_vect = TfidfVectorizer(preprocessor=preprocess_text, stop_words='english', max_features=10000)
# X_train_tfidf = tfidf_vect.fit_transform(X_train)
# visualize_feature_space(X_train_tfidf, y_train)

# # --------------------------------------
# # Define Pipelines for All Models
# # --------------------------------------
# models = {
#     'Random_Forest': RandomForestClassifier(class_weight='balanced', n_estimators=200, max_depth=20, random_state=42),
#     'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False),
#     'Logistic_Regression': LogisticRegression(penalty='l2', class_weight='balanced', random_state=42, max_iter=1000),
#     'Naive_Bayes': MultinomialNB(),
#     'Decision_Tree': DecisionTreeClassifier(random_state=42),
#     'SVM': SVC(probability=True, class_weight='balanced', random_state=42)
# }

# def create_pipeline(classifier):
#     # Use the custom preprocess function inside TfidfVectorizer
#     return Pipeline([
#         ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, stop_words='english', max_features=10000)),
#         ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
#         ('smote', SMOTE(random_state=42)),
#         ('clf', classifier)
#     ])

# pipelines = {name: create_pipeline(clf) for name, clf in models.items()}

# evaluation_results = {}
# for name, pipe in pipelines.items():
#     logger.debug("Training pipeline for model: %s", name)
#     pipe.fit(X_train, y_train)
#     y_pred = pipe.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     if hasattr(pipe.named_steps['clf'], "predict_proba"):
#         y_probs = pipe.predict_proba(X_test)[:, 1]
#         roc_auc = roc_auc_score(y_test, y_probs)
#     else:
#         roc_auc = None
#     report = classification_report(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     evaluation_results[name] = {
#         'Accuracy': accuracy,
#         'ROC-AUC Score': roc_auc,
#         'Classification Report': report,
#         'Confusion Matrix': conf_matrix
#     }
#     logger.debug("Model: %s Accuracy: %.4f", name, accuracy)
#     logger.debug("Model: %s ROC-AUC: %s", name, roc_auc)
#     logger.debug("Model: %s Classification Report:\n%s", name, report)
#     logger.debug("Model: %s Confusion Matrix:\n%s", name, conf_matrix)
#     print(f"Evaluated {name}: Accuracy = {accuracy:.4f}")

# # Select the best model (by accuracy)
# best_model_name = max(evaluation_results, key=lambda k: evaluation_results[k]['Accuracy'])
# best_pipeline = pipelines[best_model_name]
# logger.debug("Selected best model: %s", best_model_name)
# print(f"Using {best_model_name} as the best model for predictions.")

# # DEBUG: Print all vocabulary from the best pipeline's TfidfVectorizer.
# vocab = best_pipeline.named_steps['tfidf'].get_feature_names_out()
# logger.debug("Vocabulary of best pipeline (%s) has %d terms: %s", best_model_name, len(vocab), vocab)
# print(f"Vocabulary of best pipeline ({best_model_name}) has {len(vocab)} terms.")

# # --------------------------------------
# # Define Prediction Endpoint
# # --------------------------------------
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     if not data or 'text' not in data:
#         return jsonify({'error': 'Invalid request. Provide "text" field.'}), 400
    
#     input_text = data['text']
#     logger.debug("Received input text: %s", input_text)
    
#     # Debug: Tokenize input text
#     tokens = re.findall(r'\b\w+\b', input_text.lower())
#     logger.debug("Tokenized input: %s", tokens)
    
#     try:
#         prediction = best_pipeline.predict([input_text])[0]
#     except Exception as e:
#         logger.error("Error during prediction: %s", e)
#         return jsonify({'error': 'Prediction failed.'}), 500

#     label_map = {0: "Non-compliant", 1: "Compliant"}
#     predicted_label = label_map.get(prediction, "Unknown")
#     logger.debug("Predicted label: %s", predicted_label)
#     return jsonify({'prediction': predicted_label})

# if __name__ == '__main__':
#     logger.debug("Starting Flask app...")
#     app.run(debug=True)



from flask import Flask, request, jsonify
import pandas as pd
import re
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for servers
import matplotlib.pyplot as plt
from collections import Counter

# Scikit-learn & imblearn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.decomposition import TruncatedSVD

# Classifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # imblearn pipeline to work with SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --------------------------------------
# Custom Preprocess Function
# --------------------------------------
def preprocess_text(text):
    """
    Preprocess a single document: lowercases text and removes non-alphabetic characters.
    """
    text = text.lower()
    # Remove any character that is not a letter or whitespace
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# --------------------------------------
# Diagnostic Functions
# --------------------------------------
def check_duplicates(df):
    duplicate_count = df.duplicated().sum()
    print(f"Total duplicate rows in dataset: {duplicate_count}")

def analyze_token_frequencies(text_series, y):
    def get_token_counts(text_series):
        tokens = []
        for text in text_series:
            tokens.extend(re.findall(r'\b\w+\b', text.lower()))
        return Counter(tokens)
    
    tokens_class0 = get_token_counts(text_series[y == 0])
    tokens_class1 = get_token_counts(text_series[y == 1])
    print("Top 10 tokens for class 0:", tokens_class0.most_common(10))
    print("Top 10 tokens for class 1:", tokens_class1.most_common(10))

def visualize_feature_space(X, y):
    # Reduce dimensions to 2 for visualization
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_reduced = svd.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("TruncatedSVD Projection of TF-IDF Features")
    plt.colorbar(scatter, label="Class")
    plt.savefig("feature_space.png")
    plt.close()
    print("Feature space plot saved as 'feature_space.png'")

# --------------------------------------
# Load and Prepare Dataset
# --------------------------------------
df1 = pd.read_csv("gdpr_violations_noisy.csv")
df2 = pd.read_csv("gdpr_text_noisy.csv")
dataset = pd.concat([df1, df2]).drop_duplicates()
dataset['summary'] = dataset['summary'].fillna('')

check_duplicates(dataset)
X_text = dataset['summary']
y = dataset['condition']  # 0 = Non-compliant, 1 = Compliant

print("Original class distribution:")
print(y.value_counts())

analyze_token_frequencies(X_text, y)

# Split into training and test sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# Visualize feature space using TF-IDF on training data (apply custom preprocessor)
tfidf_vect = TfidfVectorizer(preprocessor=preprocess_text, stop_words='english', max_features=10000)
X_train_tfidf = tfidf_vect.fit_transform(X_train)
visualize_feature_space(X_train_tfidf, y_train)

# --------------------------------------
# Define Pipeline for XGBoost Model
# --------------------------------------
def create_pipeline(classifier):
    return Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, stop_words='english', max_features=10000)),
        ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
        ('smote', SMOTE(random_state=42)),
        ('clf', classifier)
    ])

xgboost_model = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
pipeline = create_pipeline(xgboost_model)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
if hasattr(pipeline.named_steps['clf'], "predict_proba"):
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probs)
else:
    roc_auc = None
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Evaluated XGBoost: Accuracy = {accuracy:.4f}")

vocab = pipeline.named_steps['tfidf'].get_feature_names_out()
print(f"Vocabulary of XGBoost pipeline has {len(vocab)} terms.")

# --------------------------------------
# Define Prediction Endpoint
# --------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid request. Provide "text" field.'}), 400
    
    input_text = data['text']
    
    try:
        prediction = pipeline.predict([input_text])[0]
    except Exception as e:
        logger.error("Error during prediction: %s", e)
        return jsonify({'error': 'Prediction failed.'}), 500

    label_map = {0: "Non-compliant", 1: "Compliant"}
    predicted_label = label_map.get(prediction, "Unknown")
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
