import pandas as pd

def run_backtest(df, initial_cash=100_000_000, fee=0.001):

    df = df.copy().dropna(subset=["Close","signal"])

    cash = initial_cash
    shares = 0
    position = 0  # 0 = không giữ, 1 = đang giữ

    equity_curve = []
    trades = []

    for i in range(len(df)):

        row = df.iloc[i]
        price = row["Close"]
        signal = row["signal"]

        # ===== BUY =====
        if signal == "BUY" and position == 0:
            shares = (cash * (1 - fee)) / price
            cash = 0
            position = 1

            trades.append({
                "type": "BUY",
                "price": price,
                "date": row["Date"]
            })

        # ===== SELL =====
        elif signal == "SELL" and position == 1:
            cash = shares * price * (1 - fee)
            shares = 0
            position = 0

            trades.append({
                "type": "SELL",
                "price": price,
                "date": row["Date"]
            })

        # ===== TÍNH GIÁ TRỊ =====
        total_value = cash if position == 0 else shares * price
        equity_curve.append(total_value)

    df["equity"] = equity_curve

    # ===== METRICS =====
    total_return = (equity_curve[-1] - initial_cash) / initial_cash

    max_drawdown = (
        pd.Series(equity_curve).cummax() - pd.Series(equity_curve)
    ).max()

    num_trades = len(trades) // 2

    return df, {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "num_trades": num_trades
    }
