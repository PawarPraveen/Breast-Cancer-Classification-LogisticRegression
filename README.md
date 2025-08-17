# Breast Cancer Classification - Logistic Regression

This project applies Logistic Regression to classify breast cancer tumors as **Benign** or **Malignant**.  
It demonstrates data preprocessing, model training, and evaluation using **Confusion Matrix** and **ROC Curve**.

##  Features
- Data Cleaning & Preprocessing  
- Logistic Regression Model  
- Confusion Matrix & Accuracy Evaluation  
- ROC Curve & AUC Score  

##  Tech Stack
- Python  
- scikit-learn  
- pandas  
- matplotlib / seaborn

  
## Data prep

Load dataset, basic EDA

Handle missing values (verify & impute if any)

Standardization

Multicollinearity removal via VIF

Backward elimination by p-value (statsmodels Logit)

## Models

statsmodels.Logit for inference + p-values

sklearn.LogisticRegression for regularized training + metrics

Evaluation

Confusion Matrix (with Benign/Malignant labels)

Accuracy, Precision/Recall/F1 (for Malignant)

ROC Curve + AUC


##  Results
- High accuracy with balanced precision & recall  
- ROC-AUC score close to 1.0  

