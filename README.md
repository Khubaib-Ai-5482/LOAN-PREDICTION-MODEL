# Loan Prediction Project 💰

Predict whether a person has taken a loan using personal finance data and a **Decision Tree Classifier**. This project includes preprocessing, model training, evaluation, and visualizations for insights.

---

## Overview

The dataset contains financial and demographic information for different users. The target variable is `has_loan` (0 = No Loan, 1 = Has Loan).  

The project workflow:

1. **Data Preprocessing**:  
   - Drop `user_id` (not useful for prediction)  
   - Convert categorical features (e.g., gender, education) to numeric using LabelEncoder  

2. **Train-Test Split**:  
   - 80% data for training  
   - 20% data for testing  

3. **Model Training**:  
   - Use a **Decision Tree Classifier** to predict if a user has taken a loan  

4. **Model Evaluation**:  
   - Calculate **accuracy**  
   - Generate **classification report** and **confusion matrix**  
   - Plot **ROC curve** and compute **AUC score**  

5. **Visualizations & Insights**:  
   - **Feature Importance**: See which features affect loan prediction most  
   - **Target Distribution**: Number of users with and without loans  
   - **Correlation Heatmap**: Relationships between features  

---

## Dataset Features

- `age`, `income`, `credit_score`, … (numeric features)  
- `gender`, `education`, `marital_status`, … (categorical features)  
- `has_loan` (target variable)  

> Make sure `synthetic_personal_finance_dataset.csv` is in the project folder before running the script.

---


piring AI & Data Science Engineer  
- [GitHub](https://github.com/your-username)  
- [LinkedIn](https://www.linkedin.com/in/your-linkedin/)
