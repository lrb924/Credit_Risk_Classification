# Credit Risk Classification

### Package Requirements

`pip install x` where x is the below listed packages
* numpy
* pandas
* scikit-learn
* imbalanced-learn

### Purpose of Use
* Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. This notebook uses various techniques to train and evaluate models with imbalanced classes by using the dataset contained in the Resources folder, which is a dataset of historical lending activity from a peer-to-peer lending services company, to build a model that can identify the creditworthiness of borrowers.
* The analysis contains three parts:
  * Split the Data into Training and Testing Sets
  * Create a Logistic Regression Model with the Original Data
  * Predict a Logistic Regression Model with Resampled Training Data

### Files Navigation
* Resources: Directory containing all necessary csv files
* `credit_risk_resampling.ipynb`: Notebook containing all data analysis and modeling

### Overview of Analysis

* The financial infomration used for the analysis was the lending data located in the lending_data.csv file in the Resources directory. This dataset included the features loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks, and total in debt. These features are used to determine whether the loan is healthy or high-risk (also known as the loan_status), which the models attempt to predict.
* The variables predicted were the `loan_status`, whether or not a loan was healthy or high-risk, and the `value_counts`, how many loans are considered healthy or high-risk.
* Stages of analysis:
  * Upload the data
  * Separate the data into features (X: all columns except `loan_status`) and labels (y: `loan_status`)
  * Check the balance of the labels variable
  * Split the data into training and testing datasets by using `train_test_split` function
  * Fit the logistic regression model by using the training data (X_train and y_train) and then make predictions for y (loan_status) using the X_test data
  * Evaluate the model’s performance by calculating the accuracy score of the model, generating a confusion matrix, and printing the classification report
  * Repeat the above analysis using the RandomOverSampler module from the imbalanced-learn library to resample the training data

### Results

Model 1: 
Logistic Regression Model with the Original Data:
* Accuracy Score: ~95%
* Precision:
  * Healthy loan (0): 100%
  * High-risk loan (1): 85%
* Recall: 99%

Model 2:
Logistic Regression Model with Resampled Training Data:
* Accuracy Score: ~99%
* Precision:
  * Healthy loan (0): 100%
  * High-risk loan (1): 84%
* Recall: 99%

The precision score for predicting healthy loans (0) is the same for both Model 1 and 2, and there is only a 1% difference in precision for predicting the high-risk loans (1). However, there is a higher accuracy score with Model 2.

The purpose of this analysis is to identify/predict the high-risk (1) loans, and both models had almost similar precision scores when predicing whether or not a loan was high-risk.