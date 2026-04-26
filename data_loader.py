import yfinance as yf
import pandas as pd
import concurrent.futures
import time

# =========================
# DANH SÁCH VN30
# =========================
VN30 = [
    "VCB.VN","BID.VN","CTG.VN","TCB.VN","VPB.VN",
    "FPT.VN","MWG.VN","HPG.VN","VNM.VN","VIC.VN",
    "VHM.VN","VRE.VN","SSI.VN","VND.VN","HDB.VN",
    "STB.VN","MBB.VN","ACB.VN","PNJ.VN","GAS.VN"
]

# =========================
# LOAD 1 MÃ (CÓ RETRY)
# =========================
def load_one(symbol, period="1y"):

    for attempt in range(3):  # retry 3 lần
        try:
            df = yf.download(
                symbol,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if df is None or df.empty:
                return None

            df = df.reset_index()

            # 🔥 FIX MultiIndex
            df.columns = df.columns.get_level_values(0)

            df = df[["Date","Open","High","Low","Close","Volume"]]

            df["symbol"] = symbol

            return df

        except Exception as e:
            time.sleep(1)

    return None


# =========================
# LOAD TOÀN BỘ (MULTI THREAD)
# =========================
def get_vn30_data(period="1y"):

    all_data = []

    # ⚡ chạy song song
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_one, VN30))

    for df in results:
        if df is not None:
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    data = pd.concat(all_data, ignore_index=True)

    # ép kiểu
    for col in ["Open","High","Low","Close","Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data
