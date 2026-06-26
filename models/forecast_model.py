import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import shap
import pickle

def train_forecast_model():
    df = pd.read_csv("data/processed/modeling_data.csv", parse_dates=["date"])

    features = [
        "consumer_sentiment", "disposable_income", "cpi", "unemployment",
        "grocery_delivery", "meal_kit", "food_inflation", "canned_goods",
        "sentiment_score", "promo_flag", "revenue_lag1", "revenue_lag3"
    ]
    target = "total_revenue"

    X = df[features]
    y = df[target]

    # time series cross validation — never shuffle time series data
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, preds)
        scores.append(mape)
        print(f"Fold {fold+1} MAPE: {mape:.2%}")

    print(f"Average MAPE: {np.mean(scores):.2%}")

    # final model on all data
    model.fit(X, y)
    df["predicted_revenue"] = model.predict(X)

    # SHAP values — explains which features drive each prediction
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_df = pd.DataFrame(shap_values.values, columns=features)
    shap_df["date"] = df["date"].values

    df.to_csv("data/processed/model_output.csv", index=False)
    shap_df.to_csv("data/processed/shap_values.csv", index=False)
    pickle.dump(model, open("models/forecast_model.pkl", "wb"))
    print("Model saved. Output written to data/processed/model_output.csv")
    return model, df, shap_df

if __name__ == "__main__":
    train_forecast_model()