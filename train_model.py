"""
Trains the loan approval model and saves it to disk.
This is the same pipeline built in Colab, moved here so the Flask app
does not depend on the notebook.

Run once with: python train_model.py
It creates loan_model.pkl and scaler.pkl in this folder.
"""

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

DATA_FILE = "train_u6lujuX_CVtuZ9i.csv"

FEATURE_ORDER = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
]


def load_and_clean_data():
    df = pd.read_csv(DATA_FILE)

    for col in ["Gender", "Married", "Self_Employed", "Dependents", "Credit_History"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())

    df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)
    df = df.drop("Loan_ID", axis=1)

    # Fixed encoding maps so the Flask app can reproduce them exactly.
    # This is safer than fitting LabelEncoder again elsewhere, since the
    # mapping has to stay identical between training and prediction.
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
    df["Married"] = df["Married"].map({"No": 0, "Yes": 1})
    df["Education"] = df["Education"].map({"Graduate": 0, "Not Graduate": 1})
    df["Self_Employed"] = df["Self_Employed"].map({"No": 0, "Yes": 1})
    df["Property_Area"] = df["Property_Area"].map({"Rural": 0, "Semiurban": 1, "Urban": 2})
    df["Loan_Status"] = df["Loan_Status"].map({"N": 0, "Y": 1})

    return df


def train_and_save():
    df = load_and_clean_data()

    X = df[FEATURE_ORDER]
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    print("Training accuracy:", round(train_acc, 4))
    print("Testing accuracy:", round(test_acc, 4))

    with open("loan_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Saved loan_model.pkl and scaler.pkl")


if __name__ == "__main__":
    train_and_save()
