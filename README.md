# Loan Approval Prediction

## Files
- `train_model.py` — cleans the dataset, trains the Logistic Regression model, saves `loan_model.pkl` and `scaler.pkl`
- `predictor.py` — `LoanApprovalPredictor` class: handles encoding, validation, prediction, and suggestions
- `app.py` — Flask app connecting the web page to the predictor
- `templates/index.html` — the web page (Bootstrap 5)
- `static/style.css` — custom styling
- `train_u6lujuX_CVtuZ9i.csv` — training data

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model (only needed once, or after changing the dataset):
   ```
   python train_model.py
   ```
   This creates `loan_model.pkl` and `scaler.pkl` in the same folder.

3. Start the app:
   ```
   python app.py
   ```

4. Open `http://127.0.0.1:5000` in your browser.
