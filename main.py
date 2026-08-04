from fastapi import FastAPI
import os
import pandas as pd
import numpy as np
from supabase import create_client
from xgboost import XGBClassifier

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

@app.get("/")
def home():
    return {"message": "API de predicción XGBoost lista"}

@app.get("/run-xgboost")
def run_pipeline():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"status": "error", "message": "Faltan las variables de entorno de Supabase"}
        
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1. Extracción de datos
    res_daily = supabase.table("futures_daily").select("date, close_price, volume").order("date", desc=True).limit(1000).execute()
    df_daily = pd.DataFrame(res_daily.data)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.sort_values('date', ascending=True).reset_index(drop=True)

    df_daily['close_price'] = df_daily['close_price'].astype(float)
    df_daily['volume'] = df_daily['volume'].astype(float)

    # 2. Indicadores técnicos
    df_daily['return_1d'] = df_daily['close_price'].pct_change(1)
    df_daily['return_5d'] = df_daily['close_price'].pct_change(5)
    df_daily['sma_10'] = df_daily['close_price'].rolling(window=10).mean()
    df_daily['sma_50'] = df_daily['close_price'].rolling(window=50).mean()
    df_daily['ratio_sma10'] = df_daily['close_price'] / df_daily['sma_10']

    horizon = 5
    df_daily['target'] = (df_daily['close_price'].shift(-horizon) > df_daily['close_price']).astype(int)
    df = df_daily.dropna().copy()

    features = ['close_price', 'volume', 'return_1d', 'return_5d', 'ratio_sma10']
    X = df[features]
    y = df['target']

    # 3. Entrenamiento
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X, y)

    # 4. Predicción
    latest_data = X.iloc[[-1]]
    latest_date = pd.to_datetime(df['date'].iloc[-1]).strftime('%Y-%m-%d')
    prediction = model.predict(latest_data)[0]
    prediction_proba = model.predict_proba(latest_data)[0]

    predicted_direction = 'UP' if prediction == 1 else 'DOWN'
    confidence = float(np.max(prediction_proba))

    # 5. Guardar en Supabase
    signal_payload = {
        "signal_date": latest_date,
        "predicted_direction": predicted_direction,
        "confidence_score": round(confidence, 4),
        "model_name": "XGBoost_v1"
    }

    supabase.table("market_signals").upsert(signal_payload, on_conflict="signal_date").execute()

    return {"status": "success", "signal_date": latest_date, "prediction": predicted_direction, "confidence": confidence}
