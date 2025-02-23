# RegCom & TaxRemind

**RegCom & TaxRemind** is an open-source mobile application designed for businesses and individuals who need to ensure regulatory compliance while managing critical financial deadlines. This unified solution combines:

1. **GDPR Compliance Checker:**  
   - Analyzes text input to determine whether it violates GDPR.
   - Uses a machine learning pipeline (with TF-IDF, SMOTE, and multiple classifiers) to classify text as **"Compliant"** or **"Non-compliant"**.

2. **Income Tax Payment Reminder:**  
   - Webscrapes important tax-related dates (including income tax filing, advanced tax, and TCS data) from [ClearTax](https://cleartax.in/).
   - Stores scraped dates in Supabase.
   - Runs daily scheduled jobs to send email reminders via Listmonk 15, 10, and 5 days before the due dates.

Both features are integrated into a single mobile app built with Flutter. The back-end API is built using FastAPI (and/or Flask for demonstration purposes) and secured via API keys.

---

## Problem Statement

**Problem:**  
In today's fast-paced environment, organizations and individuals face challenges in maintaining GDPR compliance and meeting critical financial deadlines, such as tax payments. Manual checks for compliance and missed tax due dates can result in severe penalties and inefficiencies.

**Solution:**  
RegCom & TaxRemind addresses these issues by providing:
- An **automated GDPR compliance checker** that quickly verifies if a text statement violates data protection regulations.
- An **income tax payment reminder system** that automatically scrapes tax due dates, stores them in a database, and sends timely email reminders.

This integrated solution simplifies regulatory oversight and financial planning through automation, reducing risk and ensuring timely compliance and payments.

---

## Features

- **GDPR Compliance Checker:**
  - Uses a custom text preprocessing function to clean input text.
  - Transforms text into TF-IDF features.
  - Applies SMOTE to balance class distribution.
  - Evaluates multiple classifiers (Random Forest, XGBoost, Logistic Regression, Naive Bayes, Decision Tree, SVM).
  - Logs detailed diagnostics including token frequencies, evaluation metrics, expected vs. actual outputs, and vocabulary of the best model.
  - Provides a `/predict` endpoint that returns a prediction.

- **Income Tax Payment Reminder:**
  - Webscrapes tax filing dates from [ClearTax](https://cleartax.in/).
  - Stores key dates (income tax, advanced tax, TCS) in Supabase.
  - Runs daily to send email reminders via Listmonk at scheduled intervals (15, 10, and 5 days before due dates).

- **Unified Mobile App:**
  - Developed in Flutter for cross-platform compatibility.
  - Connects to FastAPI endpoints (secured with API keys) for both the GDPR checker and tax reminder functionality.

---

## Technology Stack

- **Mobile App Front-end:** Flutter
- **Back-end API:** FastAPI (or Flask for demonstration)
- **Machine Learning:** scikit-learn, imbalanced-learn, XGBoost
- **Web Scraping:** Python (e.g., BeautifulSoup/Requests)
- **Database:** Supabase
- **Email Reminders:** Listmonk
- **Visualization:** Matplotlib

---

## Setup Instructions

### Prerequisites

- Python 3.11.0
- Flutter SDK
- Required Python packages:  
  ```bash
  pip install flask pandas scikit-learn imbalanced-learn xgboost matplotlib fastapi uvicorn


## **Why MIT License?**
We are using the **MIT License** instead of the **Apache License** because:
- **MIT License** is more **permissive** and allows developers to use, modify, and distribute the code with minimal restrictions.
- **Apache License** requires explicit **attribution and patent rights**, which may introduce additional legal requirements.
- The **MIT License** is **simpler and more widely used** for open-source projects, making it easier for contributors.

## **Author**
Developed by **VijayRS124**

## **License**
This project is **open-source** under the **MIT License**.

