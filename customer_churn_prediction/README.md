# Customer Churn Prediction (VS Code Starter)

This is a beginner-friendly, runnable starter project for customer churn prediction.
It is a simplified learning project, 


## What it does
- Loads customer data with pandas
- Prepares numeric and categorical features
- Trains a scikit-learn classification pipeline
- Evaluates the model with precision, recall, F1-score, ROC-AUC, and a confusion matrix
- Saves the trained model to `artifacts/churn_model.joblib`

## Run in VS Code (Windows)
1. Extract the ZIP file.
2. Open the extracted `customer_churn_prediction` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

5. Install packages:

   ```powershell
   pip install -r requirements.txt
   ```

6. Run the model:

   ```powershell
   python src/train.py
   ```

The script prints evaluation metrics and saves the model.

## Dataset note
`data/customer_churn_sample.csv` is synthetic demonstration data, not real customer records and not the IBM Telco dataset. Results on this small sample are only for learning and should not be presented as production-quality results.

## Suggested next improvements
- Replace the synthetic CSV with the IBM Telco Customer Churn dataset and document its source.
- Add stratified cross-validation and compare Logistic Regression with Random Forest.
- Add a threshold analysis and explain the precision/recall trade-off.
- Add SHAP or permutation feature importance.
- Add a Streamlit dashboard or FastAPI prediction endpoint.
