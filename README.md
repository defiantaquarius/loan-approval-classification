# Loan Approval Classification

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for predicting loan approval status. The models are trained using the Loan Approval Classification dataset and evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

The project uses the Loan Approval Classification dataset.

The dataset contains information related to loan applicants, including demographic characteristics, income, employment experience, loan characteristics, credit score, and previous loan default information.

The target variable is `loan_status`, which represents the loan approval outcome.

An 80:20 train-test split was used, with stratification to maintain the target-class distribution.

## 3. GitHub Repository

GitHub Repository Link:

https://github.com/defiantaquarius/Loan-Approval-Dataset#loan-approval-dataset

## 4. Machine Learning Models

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble)

## 5. Evaluation Metrics

The models were evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Performance

| ML Model Name        |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:---------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression  |     0.8993 | 0.9562 |      0.7888 |   0.747  | 0.7673 | 0.7036 |
| Decision Tree        |     0.8999 | 0.8558 |      0.7738 |   0.7765 | 0.7751 | 0.7108 |
| K-Nearest Neighbors  |     0.8956 | 0.9246 |      0.7909 |   0.7205 | 0.7541 | 0.6891 |
| Gaussian Naive Bayes |     0.7356 | 0.9398 |      0.4566 |   0.999  | 0.6267 | 0.5484 |
| Random Forest        |     0.9274 | 0.9741 |      0.8918 |   0.7665 | 0.8244 | 0.7826 |

## 6. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression achieved an accuracy of 0.8993, AUC of 0.9562, precision of 0.7888, recall of 0.7470, F1 score of 0.7673, and MCC of 0.7036. |
| Decision Tree | Decision Tree achieved an accuracy of 0.8999, AUC of 0.8558, precision of 0.7738, recall of 0.7765, F1 score of 0.7751, and MCC of 0.7108. |
| K-Nearest Neighbors | K-Nearest Neighbors achieved an accuracy of 0.8956, AUC of 0.9246, precision of 0.7909, recall of 0.7205, F1 score of 0.7541, and MCC of 0.6891. |
| Gaussian Naive Bayes | Gaussian Naive Bayes achieved an accuracy of 0.7356, AUC of 0.9398, precision of 0.4566, recall of 0.9990, F1 score of 0.6267, and MCC of 0.5484. |
| Random Forest | Random Forest achieved an accuracy of 0.9274, AUC of 0.9741, precision of 0.8918, recall of 0.7665, F1 score of 0.8244, and MCC of 0.7826. |
| **Overall Winner** | **Random Forest** |

## 7. Overall Winner

Based primarily on the F1 Score, the overall best-performing model is **Random Forest**.

The comparison also considers AUC and MCC to provide a broader assessment of model performance.

