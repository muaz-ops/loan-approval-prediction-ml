"""
Wraps the trained model so app.py does not need to know about encoding,
scaling, or feature order. Takes plain human-readable form values in
and returns a clean result dictionary.
"""

import pickle
import os
import pandas as pd


class LoanApprovalPredictor:

    FEATURE_ORDER = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area",
    ]

    GENDER_MAP = {"Male": 1, "Female": 0}
    YES_NO_MAP = {"Yes": 1, "No": 0}
    EDUCATION_MAP = {"Graduate": 0, "Not Graduate": 1}
    PROPERTY_MAP = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    CREDIT_MAP = {"Good": 1, "Poor": 0}

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, model_path=None, scaler_path=None):
        model_path = model_path or os.path.join(self.BASE_DIR, "loan_model.pkl")
        scaler_path = scaler_path or os.path.join(self.BASE_DIR, "scaler.pkl")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def _encode(self, form):
        """Converts raw form values (strings) into the numeric feature row
        the model expects, in the correct column order."""

        row = {
            "Gender": self.GENDER_MAP[form["gender"]],
            "Married": self.YES_NO_MAP[form["married"]],
            "Dependents": int(form["dependents"]),
            "Education": self.EDUCATION_MAP[form["education"]],
            "Self_Employed": self.YES_NO_MAP[form["self_employed"]],
            "ApplicantIncome": float(form["applicant_income"]),
            "CoapplicantIncome": float(form["coapplicant_income"]),
            "LoanAmount": float(form["loan_amount"]),
            "Loan_Amount_Term": float(form["loan_term"]),
            "Credit_History": self.CREDIT_MAP[form["credit_history"]],
            "Property_Area": self.PROPERTY_MAP[form["property_area"]],
        }
        return [row[col] for col in self.FEATURE_ORDER]

    def validate(self, form):
        """Returns a list of error messages, empty if the input is valid."""
        errors = []

        required = [
            "gender", "married", "dependents", "education", "self_employed",
            "applicant_income", "coapplicant_income", "loan_amount",
            "loan_term", "credit_history", "property_area",
        ]
        for field in required:
            if not form.get(field, "").strip():
                errors.append(f"'{field.replace('_', ' ')}' is required.")

        if errors:
            return errors

        try:
            if float(form["applicant_income"]) < 0:
                errors.append("Applicant income cannot be negative.")
            if float(form["coapplicant_income"]) < 0:
                errors.append("Coapplicant income cannot be negative.")
            if float(form["loan_amount"]) <= 0:
                errors.append("Loan amount must be greater than 0.")
            if float(form["loan_term"]) <= 0:
                errors.append("Loan term must be greater than 0.")
            if int(form["dependents"]) < 0 or int(form["dependents"]) > 3:
                errors.append("Dependents should be between 0 and 3.")
        except ValueError:
            errors.append("Income, loan amount, and term must be numbers.")

        return errors

    def predict(self, form):
        """Runs the model and returns prediction, probabilities, and a
        short suggestion based on the weakest factor in the application."""

        row = self._encode(form)
        features = pd.DataFrame([row], columns=self.FEATURE_ORDER)
        features_scaled = self.scaler.transform(features)

        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        result = {
            "approved": bool(prediction == 1),
            "label": "Approved" if prediction == 1 else "Rejected",
            "prob_approved": round(probability[1] * 100, 2),
            "prob_rejected": round(probability[0] * 100, 2),
            "suggestion": self._build_suggestion(form, prediction),
        }
        return result

    def _build_suggestion(self, form, prediction):
        """Credit history is by far the strongest factor in this dataset,
        so the suggestion focuses on that first, then falls back to
        income and loan amount."""

        if prediction == 1:
            return "This application fits the profile of approved loans in the dataset."

        if form["credit_history"] == "Poor":
            return "Credit history is the biggest factor working against this application. Applicants with a good credit history are approved far more often."

        total_income = float(form["applicant_income"]) + float(form["coapplicant_income"])
        loan_amount = float(form["loan_amount"])

        if loan_amount > 0 and total_income / loan_amount < 20:
            return "The loan amount is high relative to total income. Reducing the requested amount or adding a coapplicant's income may help."

        return "This application does not match the typical profile of approved loans in the dataset."
