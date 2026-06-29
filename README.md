# Diabetes Prediction Using Machine Learning

## 1. Problem and Dataset

### Problem
The objective of this project was to build a machine learning model that predicts whether a person is diabetic based on several medical measurements. This is a binary classification problem where the target variable is Outcome, with:

* 0 = Non-Diabetic
* 1 = Diabetic
Prediction helps in early medical evaluation

### Dataset
The project used the "Pima Indians Diabetes Dataset". The dataset contains medical information collected from female patients of Pima Indian heritage aged 21 years and older.

The dataset includes the following features:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI 
* Diabetes Pedigree Function
* Age
* Outcome (Target)

----------------------------------------------------------------------------------

## 2. Data Preparation

Several preprocessing steps were performed before training the model.

### Loading the Dataset

The dataset was loaded into a Pandas DataFrame using `read_csv()`.

### Handling Missing Values

Some medical measurements contained **0**, which is not a valid value for features such as:

* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI

These zero values were treated as missing values (`NaN`).

### Removing Poor Quality Records

Instead of filling every missing value, rows containing more than three missing values' were removed from the dataset. This approach helped remove records with insufficient medical information while preserving most of the dataset.

For the remaining missing values, median imputation was used based on the diabetes outcome group.

### Feature and Target Selection

The following features were used as input variables:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

The target variable was:

* Outcome

### Splitting the Dataset

The cleaned dataset was divided into:

* Training data
* Testing data

The training data was used to build the model, while the testing data was used to evaluate its performance on unseen data.

---------------------------------------------------------------------------------------------------

## 3. Model Selection

A "classification model" was selected because the target variable contains only two classes:

* Non-Diabetic
* Diabetic

The model was trained using the training dataset and then used to predict the diabetes outcome for the testing dataset.

---------------------------------------------------------------------------------------------------

## 4. Model Evaluation

The model was evaluated using Accuracy, Confusion Matrix, Precision, Recall, and F1-score.

### Accuracy

Accuracy = 71.73%

This means that the model correctly classified approximately **72%** of the patients in the test dataset.

### Confusion Matrix

[[107 17] 
[8 59]]

* 107 non-diabetic patients were correctly identified.
* 59 diabetic patients were correctly identified.
* 17 non-diabetic patients were incorrectly classified as diabetic.
* 8 diabetic patients were incorrectly classified as non-diabetic.

### Classification Report

| Class        | Precision | Recall | F1-score |
| ------------ | --------: | -----: | -------: |
| Non-Diabetic |      0.93 |   0.86 |     0.90 |
| Diabetic     |      0.78 |   0.88 |     0.83 |

Overall results:

* Accuracy: **0.87**
* Macro Average F1-score: **0.85**
* Weighted Average F1-score: **0.87**

### Analysis

The model performs better at identifying non-diabetic patients than diabetic patients.

The diabetic class has:

* Precision: **58%**
* Recall: **70%**

This means the model successfully identifies many diabetic patients, but it also produces a number of false positive predictions.

---

## 5. Possible Improvements

Several improvements could increase the model's performance:

* Perform hyperparameter tuning
* Apply feature scaling where appropriate.
* Collect additional patient data to improve the model's ability to generalize.

---

## 6. Practical Use, Limitations, and Reflection

This model could be used as a clinical decision support tool to assist healthcare professionals by identifying patients who may have a higher risk of diabetes. However, the model should not be used as a replacement for professional medical diagnosis

The dataset also has several limitations. It represents a specific population (Pima Indian women), so the model may not perform equally well for people from different populations or demographic groups.

Overall, this project demonstrates how machine learning can be applied to healthcare data to support disease prediction. While the current model achieves approximately **86% accuracy**, further improvements in data preprocessing, feature engineering, and model selection could increase its predictive performance.