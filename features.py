import pandas as pd
import numpy as np


# =========================
# ADD INDICATORS
# =========================
def add_indicators(df):
    df = df.copy()
    # Chuẩn hóa tên cột
    if 'close' in df.columns: df = df.rename(columns={'close': 'Close'})
    
    close = df["Close"]
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df.dropna()

def segment_customer(df):
    vol = df["Close"].pct_change().std()
    return "Aggressive" if vol > 0.02 else "Conservative"


# =========================
# TARGET (FIX IMBALANCE)
# =========================
def create_target(df):

    df = df.copy()

    # 🔥 chỉ cần tăng nhẹ là đủ (tránh imbalance)
    df["target"] = (
        df.groupby("symbol")["Close"]
        .shift(-1) > df["Close"]
    ).astype(int)

    return df


# =========================
# SIGNAL
# =========================
def trading_signals(df):

    df = df.copy()

    df["signal"] = "HOLD"

    # 🔥 nới điều kiện để có giao dịch
    df.loc[df["rsi"] < 45, "signal"] = "BUY"
    df.loc[df["rsi"] > 60, "signal"] = "SELL"

    # ===== STOP LOSS / TAKE PROFIT =====
    df["stop_loss"] = df["Close"] * 0.95
    df["take_profit"] = df["Close"] * 1.08

    return df


# =========================
# DEBUG DATA (RẤT HỮU ÍCH)
# =========================
def check_data_quality(df):

    report = {
        "num_rows": len(df),
        "num_symbols": df["symbol"].nunique(),
        "missing_close": df["Close"].isna().sum(),
        "target_distribution": df["target"].value_counts(normalize=True).to_dict()
    }

    return report


#Khuyến nghị
def get_detailed_advice(df_symbol):
    """
    Trả về lời khuyên theo cấu trúc: Đối tượng -> Hành động -> Lý do
    """
    if df_symbol.empty or len(df_symbol) < 2:
        return None

    latest = df_symbol.iloc[-1]
    close = latest['Close']
    ma20 = latest['ma20']
    rsi = latest['rsi']
    
    # Phân tích xu hướng biểu đồ
    is_uptrend = close > ma20
    is_oversold = rsi < 35
    is_overbought = rsi > 65

    advice_data = {}

    # 1. Đối tượng: Aggressive (Mạo hiểm)
    if is_oversold:
        advice_data["Aggressive"] = {
            "action": "MUA (Bắt đáy)",
            "reason": f"Chỉ số RSI ở mức thấp ({rsi:.1f}), biểu đồ cho thấy giá đã giảm quá sâu vào vùng quá bán. Cơ hội ăn nhịp hồi kỹ thuật cực cao."
        }
    elif is_overbought:
        advice_data["Aggressive"] = {
            "action": "BÁN (Chốt lời)",
            "reason": f"RSI đã vượt ngưỡng 65 ({rsi:.1f}), giá đang quá nóng. Nên bán để hiện thực hóa lợi nhuận trước khi có nhịp điều chỉnh."
        }
    else:
        advice_data["Aggressive"] = {
            "action": "GIỮ",
            "reason": "Giá đang trong vùng tích lũy ổn định, chưa có tín hiệu bứt phá rõ rệt để vào lệnh mới."
        }

    # 2. Đối tượng: Conservative (Thận trọng)
    if is_uptrend and rsi < 60:
        advice_data["Conservative"] = {
            "action": "MUA / GIỮ",
            "reason": f"Giá ({close:,.0f}) nằm trên đường MA20 ({ma20:,.0f}), xác lập xu hướng tăng bền vững. Rủi ro thấp, an toàn để nắm giữ dài hạn."
        }
    elif not is_uptrend:
        advice_data["Conservative"] = {
            "action": "BÁN / ĐỨNG NGOÀI",
            "reason": f"Giá đã thủng đường trung bình MA20. Biểu đồ chuyển sang xu hướng giảm, nên giữ tiền mặt để bảo toàn vốn."
        }
    else:
        advice_data["Conservative"] = {
            "action": "GIỮ / THEO DÕI",
            "reason": "Xu hướng vẫn ổn nhưng giá đang tiến gần vùng cản, không nên mua đuổi lúc này."
        }

    return advice_data


def segment_customer(df):
    """Phân loại nhóm khách hàng dựa trên độ biến động thực tế."""
    if df.empty or len(df) < 10:
        return "Conservative"
    # Dùng cột 'Close' vì yfinance trả về tên này
    volatility = df["Close"].pct_change().std()
    return "Aggressive" if volatility > 0.02 else "Conservative"

def get_detailed_recommendation(df):
    df = df.copy().dropna()

    if len(df) < 2:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # ===== BASIC =====
    price = latest["Close"]
    rsi = latest["rsi"]
    signal = latest["signal"]
    ma20 = latest["ma20"]
    ma50 = latest["ma50"]

    # ===== TREND =====
    if price > ma20 > ma50:
        trend = "Tăng mạnh"
    elif price < ma20 < ma50:
        trend = "Giảm"
    else:
        trend = "Sideway"

    # ===== PRICE CHANGE =====
    price_change = ((price - prev["Close"]) / prev["Close"]) * 100

    # ===== RSI STATUS =====
    if rsi < 30:
        rsi_status = "Quá bán (Rẻ)"
    elif rsi > 70:
        rsi_status = "Quá mua (Đắt)"
    else:
        rsi_status = "Ổn định"

    # ===== MOMENTUM =====
    momentum = "MẠNH" if ma20 > ma50 else "YẾU"

    # ===== SIGNAL ACTION =====
    if signal == "BUY":
        action = "NÊN MUA"
    elif signal == "SELL":
        action = "NÊN BÁN"
    else:
        action = "CHỜ"

    # ===== PRICE ZONE =====
    stop_loss = round(price * 0.95, 2)
    take_profit = round(price * 1.08, 2)

    # ===== ANALYSIS TEXT =====
    price_status = f"{'📈' if price_change >= 0 else '📉'} Giá {trend} ({round(price_change,2)}%)"

    indicator_reason = (
        f"RSI = {round(rsi,1)} ({rsi_status}). "
        f"Xu hướng MA: {momentum}."
    )

    # ===== ADVICE =====
    advice = {}

    # 🔥 Aggressive
    if rsi < 35:
        advice["Aggressive"] = "🔥 MUA (bắt đáy ngắn hạn)"
    elif rsi > 65:
        advice["Aggressive"] = "🔔 BÁN (chốt lời nhanh)"
    else:
        advice["Aggressive"] = "💎 GIỮ (chờ breakout)"

    # 🛡 Conservative
    if price < ma50:
        advice["Conservative"] = "🛡️ TRÁNH / GIẢM TỶ TRỌNG"
    elif price > ma50 and momentum == "MẠNH":
        advice["Conservative"] = "✅ GIỮ / MUA DÀI HẠN"
    else:
        advice["Conservative"] = "🧱 GIỮ QUAN SÁT"

    # ===== FINAL RETURN =====
    return {
        # dữ liệu số
        "price": round(price, 2),
        "rsi": round(rsi, 1),
        "trend": trend,
        "action": action,
        "stop_loss": stop_loss,
        "take_profit": take_profit,

        # phân tích
        "price_status": price_status,
        "indicator_reason": indicator_reason,

        # lời khuyên
        "advice": advice
    }

import pandas as pd

def get_top_recommendations(data):
    results = []

    symbols = data["symbol"].unique()

    for sym in symbols:
        df = data[data["symbol"] == sym].copy()

        df = df.dropna(subset=["Close", "ma20", "ma50", "rsi", "target"])

        if len(df) < 50:
            continue

        latest = df.iloc[-1]

        score = 0

        # ===== 1. RSI (định giá) =====
        if latest["rsi"] < 35:
            score += 2   # quá bán → tốt
        elif latest["rsi"] < 50:
            score += 1

        # ===== 2. TREND =====
        if latest["Close"] > latest["ma20"] > latest["ma50"]:
            score += 2   # trend mạnh
        elif latest["Close"] > latest["ma20"]:
            score += 1

        # ===== 3. SIGNAL =====
        if latest.get("signal") == "BUY":
            score += 2

        # ===== 4. MOMENTUM =====
        if latest["ma20"] > latest["ma50"]:
            score += 1

        # ===== 5. BONUS (RSI đẹp + trend) =====
        if 30 < latest["rsi"] < 50 and latest["Close"] > latest["ma20"]:
            score += 1

        results.append({
            "symbol": sym,
            "price": round(latest["Close"], 2),
            "rsi": round(latest["rsi"], 1),
            "trend": "UP" if latest["Close"] > latest["ma20"] else "DOWN",
            "signal": latest.get("signal", "HOLD"),
            "score": score
        })

    if not results:
        return pd.DataFrame()

    df_result = pd.DataFrame(results)

    # ===== SORT =====
    df_result = df_result.sort_values(by="score", ascending=False)

    # ===== TOP 5 =====
    return df_result.head(5)

