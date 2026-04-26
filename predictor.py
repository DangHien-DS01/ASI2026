import pandas as pd

# =========================
# LEVEL 1: AI MODEL
# =========================
def predict_with_model(df, model):
    try:
        latest = df.iloc[-1:]

        pred = model.predict(latest[["ma20","ma50","rsi"]])[0]

        return {
            "prediction": "UP" if pred == 1 else "DOWN",
            "source": "AI Model"
        }
    except:
        return None


# =========================
# LEVEL 2: RULE-BASED
# =========================
def predict_with_rules(df):
    latest = df.iloc[-1]

    if latest["rsi"] < 35 and latest["Close"] > latest["ma20"]:
        return {"prediction": "UP", "source": "RSI + MA"}

    if latest["rsi"] > 65:
        return {"prediction": "DOWN", "source": "RSI Overbought"}

    if latest["Close"] > latest["ma20"]:
        return {"prediction": "UP", "source": "MA Trend"}

    return {"prediction": "DOWN", "source": "Weak Trend"}


# =========================
# LEVEL 3: SIMPLE TREND
# =========================
def predict_simple(df):
    if len(df) < 2:
        return {"prediction": "UNKNOWN", "source": "No Data"}

    latest = df.iloc[-1]["Close"]
    prev = df.iloc[-2]["Close"]

    if latest > prev:
        return {"prediction": "UP", "source": "Price Momentum"}
    else:
        return {"prediction": "DOWN", "source": "Price Momentum"}


# =========================
# MAIN FUNCTION
# =========================
import numpy as np

def smart_predict(df, model):

    latest = df.dropna().iloc[-1]

    # ❗ nếu không có model → fallback
    if model is None:
        return {
            "prediction": "UP" if latest["rsi"] < 40 else "DOWN",
            "source": "RSI fallback"
        }

    try:
        pred = model.predict([[
            latest["ma20"],
            latest["ma50"],
            latest["rsi"]
        ]])[0]

        return {
            "prediction": "UP" if pred == 1 else "DOWN",
            "source": "AI model"
        }

    except:
        return {
            "prediction": "UP" if latest["ma20"] > latest["ma50"] else "DOWN",
            "source": "MA fallback"
        }

