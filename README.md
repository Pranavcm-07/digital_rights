# **GDPR Compliance Classification with K-Fold Cross-Validation**

## **Overview**
This project aims to classify **GDPR compliance** using machine learning techniques. It implements **K-Fold Cross-Validation** with multiple classifiers, including **Support Vector Machines (SVM), Logistic Regression, and Naïve Bayes**, to evaluate model performance.

The dataset consists of GDPR-related text data categorized as either **"compliance" or "violation."** Data preprocessing includes **text vectorization, feature selection, and handling class imbalance using SMOTE.**

## **Features**
- **Text Preprocessing:** Removes duplicates and applies **TF-IDF vectorization**.
- **Data Augmentation:** Uses **synonym replacement** for better generalization.
- **Feature Selection:** Uses the **Chi-Square test** to select the top features.
- **Class Imbalance Handling:** Applies **SMOTE** to balance the dataset.
- **Model Evaluation:** Uses **Stratified K-Fold Cross-Validation**.

## **Installation**
To run this project, install the necessary dependencies:

```bash
pip install numpy pandas scikit-learn nltk imbalanced-learn
```

Additionally, download the necessary **NLTK data**:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## **Usage**
1. **Prepare the dataset**: Place the GDPR dataset (`gdpr_dataset.csv`) in the project directory.
2. **Run the script**: Execute the Python script to train models and evaluate performance using **K-Fold Cross-Validation**.

```bash
python kfold_gdpr.py
```

## **Expected Output**
The script prints the **K-Fold accuracy scores** for each classifier:
```
K-Fold Cross-Validation with SVM
Fold Accuracy: 0.95
...
Average Accuracy (SVM): 0.96

K-Fold Cross-Validation with Logistic Regression
Fold Accuracy: 0.93
...
Average Accuracy (Logistic Regression): 0.94
```

## **Future Enhancements**
- Implement **deep learning models** for better classification.
- Use advanced **NLP techniques like BERT** for feature extraction.
- Expand the **dataset** for improved generalization.

## **Why MIT License?**
We are using the **MIT License** instead of the **Apache License** because:
- **MIT License** is more **permissive** and allows developers to use, modify, and distribute the code with minimal restrictions.
- **Apache License** requires explicit **attribution and patent rights**, which may introduce additional legal requirements.
- The **MIT License** is **simpler and more widely used** for open-source projects, making it easier for contributors.

## **Author**
Developed by **[Your Name]**

## **License**
This project is **open-source** under the **MIT License**.

