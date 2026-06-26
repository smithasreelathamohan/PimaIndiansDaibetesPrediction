#%% #Step1: Import libraries
print("\n----------------------------------")
print(" Import libraries and set filepath")
print("----------------------------------")
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
print("Imported.....")

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DATA_RAW = "diabetes.csv"
DATA_CLEAN = "diabetes_cleaned.csv"
PLOT_BEFORE = "barchart_basedon_outcomes.png"
PREDICTION_RESULT = "diabetese_prediction_result.csv"
EVALUATION_REPORT = "model_evaluation.txt"
TREE_RULE = "diabetes_prediction_tree_rules.txt"
print("Path set.....")
print("Done.....")

#%% #Step2: Load from CSV
print("\n----------------------------------")
print("        Load and analyse data")
print("----------------------------------")
df = pd.read_csv(DATA_DIR / DATA_RAW)

print("\nDaibetes Dataset (First five rows):")
print("-------------------------------------")
print(df.head())

print("\nDataset info:")
print("-----------------")
print(df.info())

print("\nDataset describe:")
print("-----------------")
print(df.describe())
print("\nDone.........")



# %% #Step3: Clean Dataset
print("\n----------------------------------")
print("     Cleaning data")
print("----------------------------------")

print(f"Column: {list(df.columns)}")
print("\nMissing values :")
print("---------------------------")
print(df.isna().sum())


print("\nRows with 0 in more than 2 column")
print("----------------------------------")
zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
# drop rows with more than 3 0 values, it can be crapy row 
zero_counts_per_row = (df[zero_as_missing_cols] == 0).sum(axis=1)
rows_to_drop = zero_counts_per_row > 3


print(f"\nRows with more than 3 zero values in columns (to be dropped): {rows_to_drop.sum()}")
df = df[~rows_to_drop].reset_index(drop=True)
print(f"Shape after dropping those rows: {df.shape}")

# after droping rows with more than 3 0 values
print("\nColumns with unrealistic 0 values left:")
print("----------------------------------")
for col in zero_as_missing_cols:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} / {len(df)} entries")


# replace 0 with NaN
df_clean = df.copy()
df_clean[zero_as_missing_cols] = df_clean[zero_as_missing_cols].replace(0, np.nan)

print("\nAfter replacing 0 with NaN:")
print("----------------------------------")
for col in zero_as_missing_cols:
    zero_count = (df_clean[col] == 0).sum()
    print(f"{col}: {zero_count} / {len(df)} entries")

print("\nMissing values after converting 0 -> NaN")
print("------------------------------------------")
print(df_clean.isnull().sum())

print("\nFilling with median considering outcome column")
print("------------------------------------------")
def fill_with_median(x):
    """Fill missing values in a column with that column's own median."""
    return x.fillna(x.median())

for col in zero_as_missing_cols:
    df_clean[col] = df_clean.groupby('Outcome')[col].transform(fill_with_median)

print("\nConvert to int for some columns")
print("------------------------------------------")
int_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin"]
for col in int_cols:
    df_clean[col] = df_clean[col].round().astype(int)

print("\nMissing values after adding median:")
print("------------------------------------------")
print(df_clean.isnull().sum())

print("\nCleaned Dataset info:")
print("-----------------")
print(df_clean.info())

print("\nDescribe Cleaned data:")
print("------------------------------------------")
print(df_clean.describe())

df_clean.to_csv(DATA_DIR / DATA_CLEAN, index=False)
print(f"\nCleaned dataset saved to: {DATA_DIR / DATA_CLEAN}")
print("\nDone.....")


# %% Step4: Visualise Data set
print("\n----------------------------------")
print("          Visualise Data Set")
print("----------------------------------")
OUTPUT_DIR.mkdir(exist_ok=True)

features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

averages_by_outcome = df_clean.groupby("Outcome")[features].mean()
 
#plot two chart side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 8))
 
# Left chart: Non-Diabetic (Outcome = 0)
averages_by_outcome.loc[0].plot(kind="bar", ax=axes[0], color="blue")
axes[0].set_title("Non-Diabetic(0): Average Value per Feature")
axes[0].set_ylabel("Average Value")
axes[0].tick_params(axis="x", rotation=45)
 
# Right chart: Diabetic (Outcome = 1)
averages_by_outcome.loc[1].plot(kind="bar", ax=axes[1], color="red")
axes[1].set_title("Diabetic(1): Average Value per Feature")
axes[1].set_ylabel("Average Value")
axes[1].tick_params(axis="x", rotation=45)
 
plt.suptitle("Average Feature Values: Non-Diabetic(0) vs Diabetic(1)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / PLOT_BEFORE, dpi=150, bbox_inches="tight")
print("Saved bar plot before")
plt.show()

print("\nDone.....")


# %% Step5: Prepare Feature and Target
print("\n------------------------------------")
print("Prepare Feature/Target and split data")
print("--------------------------------------")

feature_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

target_column = "Outcome"

X = df[feature_columns]  # X usually means the input columns in machine learning examples
y = df[target_column]  # y usually means the value we want to predict

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)
print("\nDone......")



# %% Step6: Train Decision Tree
print("\n----------------------------------")
print("      Train Decision Tree")
print("----------------------------------")

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("\nDone......")

# %% Step7: Predict and Save Results
print("\n---------------------------------------------------")
print("      Predict and Save Results")
print("-----------------------------------------------------")

predictions = model.predict(X_test)

results_df = X_test.copy()
results_df["actual_outcome"] = y_test.values
results_df["predicted_outcome"] = predictions

results_df.to_csv(OUTPUT_DIR / PREDICTION_RESULT, index=False)
print("\nPrediction result saved.....")

print("\nPrediction Results comparision on test data: ")
print(results_df.to_string(index=False))
print("\nDone...........")

# %% Step8: Evaluate: Check Accuracy and Confusion metrix
print("\n---------------------------------------------------")
print("      Evaluate: Check Accuracy and Confusion metrix")
print("-----------------------------------------------------")

label_order = [0, 1]

accuracy = accuracy_score(y_test, predictions)
matrix = confusion_matrix(y_test, predictions, labels=label_order)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(matrix)

cm = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

report = classification_report(y_test, predictions, labels=label_order)

print("\n--- Classification report ---")
print(report)

with open(OUTPUT_DIR / EVALUATION_REPORT, "w") as file:
    file.write("-----------------------------------------\n")
    file.write("     Accuracy and Confusion metrix\n")
    file.write("-----------------------------------------\n")
    file.write(f"\nAccuracy: {accuracy}\n")
    file.write(f"\nConfusion Matrix: \n{matrix}\n")
    file.write(f"\nClassificatio Report: ")
    file.write(report)

print("Classification report saved successfully.")
print("\nDone....")

#%% Step9: Export the tree rule as text
print("\n---------------------------------------------------")
print("      Export the tree rule ")
print("-----------------------------------------------------")
tree_rules = export_text(model, feature_names=feature_columns)

with open(OUTPUT_DIR / TREE_RULE, "w", encoding="utf-8") as file:
    file.write(tree_rules)

print("\n--- Tree rules ---")
print(tree_rules)
print("Saved diabetese prediction tree rule")
print("Done....")

#%% Step10: Predict New Case
print("\n---------------------------------------------------")
print("      Predict New Case ")
print("-----------------------------------------------------")
new_record = pd.DataFrame(
    {
        "Pregnancies": [0,3,1,5,2],
        "Glucose": [140,110,80,112,130],
        "BloodPressure": [66,74,90,69,92],
        "SkinThickness": [35,19,48,36,40],
        "Insulin": [180,88,100,225,333],
        "BMI": [34.2,22.3,40.5,33.2,38.4],
        "DiabetesPedigreeFunction": [1.351,0.645,0.734,1.298,0.345],
        "Age": [30,44,29,56,33]

    }
)

new_predictions = model.predict(new_record)

print("\n--- New predictions ---")
print("New predictions:", new_predictions)

results = new_record.copy()
results["Predicted_Outcome"] = new_predictions

print(results.to_string(index=False))

print("Done....")