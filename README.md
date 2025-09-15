# Fraud Detection Model — Financial Transactions

This project develops a **machine learning model** to proactively detect fraudulent transactions for a financial company.  
It uses a large dataset (6.3M+ rows) of transactions and applies data cleaning, feature engineering, model training, and evaluation to build an effective fraud detection pipeline.

---

## 📂 Project Structure
├── fraud_detection_case.ipynb # Jupyter Notebook with full analysis & modeling
├── Fraud.csv # Dataset (not included here due to size)
├── README.md # Project documentation


---

## 🎯 Business Objective
The goal is to:
- Detect fraudulent transactions with **high recall** (catch as many frauds as possible)
- Minimize **false positives** (reduce unnecessary investigation/manual review)
- Provide **actionable insights** to strengthen fraud prevention measures.

---

## 🧠 Approach
1. **Data Cleaning**
   - Handle missing values, outliers, and incorrect types
   - Create derived features such as balance mismatches (`delta_orig`, `delta_dest`)
   - Remove or cap extreme values to make training stable
   - Check for multicollinearity (VIF analysis)

2. **Feature Engineering**
   - Transaction amount transformations (`log`, capped)
   - Balance mismatch flags
   - Merchant/non-merchant indicators
   - One-hot encoding for transaction types
   - Large transfer rule-based flag

3. **Model Development**
   - **Random Forest Classifier** (balanced class weights)
   - **Logistic Regression** (baseline for interpretability)

4. **Model Evaluation**
   - ROC-AUC, Precision-Recall curve (focus on imbalanced dataset)
   - Optimal threshold selection (maximize F1 or target recall)
   - Confusion matrix, classification report

5. **Insights**
   - High transaction amount and balance mismatches are strong indicators of fraud
   - Certain transaction types (e.g., TRANSFER) are more fraud-prone
   - Original balance and destination account behavior are important predictors

---

## 📊 Example Outputs

- **Precision-Recall Curve:** Helps select operating threshold based on desired recall.
- **Feature Importances:** Shows which features drive fraud predictions.
- **Classification Report:** Precision, recall, F1-score for model performance.

---

## 🛠️ Setup & Installation

### 1. Clone the Repository

git clone <your-repo-url>
cd fraud-detection

### 2. Install Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels lightgbm jupyter ipywidgets

### 3. Run Jupyter Notebook
jupyter notebook

### Metrics to Monitor Post-Deployment

Fraud detection recall (catch rate)
Precision (false positive rate)
Total prevented fraud losses
Manual review workload
Model drift (feature distribution monitoring)
