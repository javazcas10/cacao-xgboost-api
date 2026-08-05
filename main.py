import os
import pandas as pd
import numpy as np
from fastapi import FastAPI
from supabase import create_client
from xgboost import XGBClassifier
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

@app.get("/")
def home():
    return {"message": "API XGBoost + Forecast lista"}

@app.get("/run-xgboost")
def run_pipeline():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"status": "error", "message": "Faltan las variables de entorno de Supabase"}
        
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1. Extracción de datos
    res_daily = supabase.table("futures_daily").select("date, close_price, volume").order("date", desc=True).limit(500).execute()
    df_daily = pd.DataFrame(res_daily.data)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.sort_values('date', ascending=True).reset_index(drop=True)

    df_daily['close_price'] = df_daily['close_price'].astype(float)
    df_daily['volume'] = df_daily['volume'].astype(float)

    # ==================== 2. MODELO 1: XGBOOST ====================
    df_xgb = df_daily.copy()
    df_xgb['return_1d'] = df_xgb['close_price'].pct_change(1)
    df_xgb['return_5d'] = df_xgb['close_price'].pct_change(5)
    df_xgb['sma_10'] = df_xgb['close_price'].rolling(window=10).mean()
    df_xgb['ratio_sma10'] = df_xgb['close_price'] / df_xgb['sma_10']

    horizon = 5
    df_xgb['target'] = (df_xgb['close_price'].shift(-horizon) > df_xgb['close_price']).astype(int)
    df_xgb = df_xgb.dropna().copy()

    features = ['close_price', 'volume', 'return_1d', 'return_5d', 'ratio_sma10']
    X = df_xgb[features]
    y = df_xgb['target']

    xgb_model = XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
    xgb_model.fit(X, y)

    latest_data = X.iloc[[-1]]
    latest_date = pd.to_datetime(df_xgb['date'].iloc[-1]).strftime('%Y-%m-%d')
    prediction = xgb_model.predict(latest_data)[0]
    prediction_proba = xgb_model.predict_proba(latest_data)[0]

    predicted_direction = 'UP' if prediction == 1 else 'DOWN'
    confidence = float(np.max(prediction_proba))

    signal_payload = {
        "signal_date": latest_date,
        "predicted_direction": predicted_direction,
        "confidence_score": round(confidence, 4),
        "model_name": "XGBoost_v1"
    }
    supabase.table("market_signals").upsert(signal_payload, on_conflict="signal_date").execute()

    # ==================== 3. MODELO 2: FORECAST 5 DÍAS (Holt-Winters) ====================
    model_hw = ExponentialSmoothing(df_daily['close_price'], trend='add', seasonal=None).fit()
    forecast_prices = model_hw.forecast(5).values

    # Guardar los 5 días proyectados en Supabase
    forecast_records = []
    base_date = pd.to_datetime(latest_date)
    for i, p in enumerate(forecast_prices):
        future_date = (base_date + pd.BDay(i + 1)).strftime('%Y-%m-%d')
        forecast_records.append({
            "base_date": latest_date,
            "forecast_date": future_date,
            "predicted_price": round(float(p), 2),
            "step": i + 1
        })

    supabase.table("price_forecasts").upsert(forecast_records, on_conflict="base_date,forecast_date").execute()

    return {
        "status": "success",
        "signal_date": latest_date,
        "xgb_prediction": predicted_direction,
        "confidence": confidence,
        "lstm_forecast": [round(float(p), 2) for p in forecast_prices]
    }
