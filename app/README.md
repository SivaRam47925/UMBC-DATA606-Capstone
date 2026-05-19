# Streamlit Application

This folder contains the interactive Streamlit dashboard for ICU mortality prediction.

## Files

- `app.py` - Main Streamlit dashboard
- `final_16_features_plus_target.csv` - Final dataset used by the app
- `final_gradient_boosting_model.pkl` - Trained Gradient Boosting model
- `model_imputer.pkl` - Median imputer for missing values
- `model_features.pkl` - Feature order used during prediction
- `train_local_model.py` - Script to retrain and save model artifacts
- `requirements.txt` - Required Python packages

## Run the App

```bash
python -m streamlit run app.py
