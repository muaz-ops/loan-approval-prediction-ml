from flask import Flask, render_template, request
from predictor import LoanApprovalPredictor

app = Flask(__name__)
predictor = LoanApprovalPredictor()

FORM_DEFAULTS = {
    "gender": "", "married": "", "dependents": "", "education": "",
    "self_employed": "", "applicant_income": "", "coapplicant_income": "",
    "loan_amount": "", "loan_term": "", "credit_history": "", "property_area": "",
}


@app.route("/")
def home():
    return render_template("index.html", form=FORM_DEFAULTS, result=None, errors=None)


@app.route("/predict", methods=["POST"])
def predict():
    form = {key: request.form.get(key, "") for key in FORM_DEFAULTS}

    errors = predictor.validate(form)
    if errors:
        return render_template("index.html", form=form, result=None, errors=errors)

    result = predictor.predict(form)
    return render_template("index.html", form=form, result=result, errors=None)


if __name__ == "__main__":
    app.run(debug=True)
