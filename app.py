import streamlit as st

@st.cache_data(ttl=3600)
def load_data_cached():
    return get_vn30_data()

import pandas as pd
import numpy as np

from data_loader import get_vn30_data
from features import (
    add_indicators,
    create_target,
    trading_signals,
    get_detailed_recommendation,
    get_top_recommendations
)
from model import train_model
from predictor import smart_predict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("Hệ Thống Dự Báo Thị Trường Chứng Khoán - ASI")

# =========================
# LOAD DATA
# =========================
if "data" not in st.session_state:
    st.session_state.data = None

if st.button("📥 Tải Dữ Liệu VN30"):
    with st.spinner("Đang tải..."):
        data = get_vn30_data()

        if data is None or data.empty:
            st.error("❌ Không load được dữ liệu")
            st.stop()

        data = add_indicators(data)
        data = create_target(data)
        data = trading_signals(data)

        st.session_state.data = data
        st.success("✅ Load Xong!")

if st.session_state.data is None:
    st.warning("👉 Nhấn 'Tải Dữ Liệu VN30' Để Tải Dữ Liệu!")
    st.stop()

data = st.session_state.data

# =========================
# SELECT STOCK
# =========================
symbols = sorted(data["symbol"].unique())
selected = st.selectbox("📌 Chọn Mã Cổ Phiếu", symbols)

df = data[data["symbol"] == selected].copy()

if df.empty:
    st.warning("Không có dữ liệu")
    st.stop()

# =========================
# CHART
# =========================
st.subheader("📈 Biểu Đồ Xu Hướng Giá Cổ Phiếu")

chart_df = df.set_index("Date")[["Close", "ma20", "ma50"]]
st.line_chart(chart_df)

# =========================
# METRICS
# =========================
st.subheader("📊 Phân Tích Hành Vi Cổ Phiếu")

df_model = df.dropna(subset=["ma20","ma50","rsi","target"]).copy()

if len(df_model) > 30:

    X = df_model[["ma20","ma50","rsi"]]
    y = df_model["target"]

    # ❗ CHECK DATA CÓ ĐỦ 2 CLASS KHÔNG
    if len(set(y)) < 2:
        st.warning("⚠️ Dữ liệu chỉ có 1 trạng thái (toàn tăng hoặc toàn giảm) → không thể đánh giá model")
        st.stop()

    # TRAIN MODEL
    model = train_model(df_model)

    preds = model.predict(X)

    # ❗ CHECK MODEL CÓ ĐOÁN 1 CHIỀU KHÔNG
    one_side_prediction = len(set(preds)) == 1

    # ===== METRICS =====
    acc = accuracy_score(y, preds)

    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    # ===== AUC =====
    auc = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:,1]
            auc = roc_auc_score(y, probs)
    except:
        auc = None

    # ===== DIỄN GIẢI THÔNG MINH =====

    # Accuracy
    if acc >= 0.75:
        acc_text = "Xu hướng giá rất rõ, cổ phiếu dễ dự đoán"
    elif acc >= 0.65:
        acc_text = "Có xu hướng nhưng vẫn nhiễu"
    else:
        acc_text = "Giá biến động khó lường, thị trường không ổn định"

    # Precision
    if precision >= 0.7:
        precision_text = "Tín hiệu MUA đáng tin, ít mua nhầm"
    elif precision >= 0.5:
        precision_text = "Tín hiệu MUA tạm ổn nhưng cần lọc thêm"
    else:
        precision_text = "Dễ mua nhầm cổ phiếu xấu"

    # Recall
    if recall >= 0.7:
        recall_text = "Bắt sóng tăng tốt, ít bỏ lỡ cơ hội"
    elif recall >= 0.5:
        recall_text = "Bắt được một phần sóng tăng"
    else:
        recall_text = "Dễ bỏ lỡ các nhịp tăng mạnh"

    # F1
    if f1 >= 0.7:
        f1_text = "Tín hiệu giao dịch ổn định"
    elif f1 >= 0.5:
        f1_text = "Tín hiệu ở mức trung bình"
    else:
        f1_text = "Tín hiệu yếu, khó sử dụng"

    # AUC
    if auc is None:
        auc_text = "Không đủ dữ liệu xác suất"
    elif auc >= 0.75:
        auc_text = "Phân biệt rõ xu hướng tăng hoặc giảm"
    elif auc >= 0.6:
        auc_text = "Xu hướng có nhưng còn nhiễu"
    else:
        auc_text = "Thị trường nhiễu mạnh, khó phân tích"

    # ===== DATAFRAME =====
    metrics_df = pd.DataFrame({

        "Chỉ số": ["Accuracy","Precision","Recall","F1-score","AUC-ROC"],

        "Giá trị": [
            round(acc,3),
            round(precision,3),
            round(recall,3),
            round(f1,3),
            round(auc,3) if auc is not None else "N/A"
        ],

        "Ý nghĩa với cổ phiếu": [
            acc_text,
            precision_text,
            recall_text,
            f1_text,
            auc_text
        ]
    })

    st.dataframe(metrics_df, use_container_width=True)

    # ===== CẢNH BÁO THÔNG MINH =====
    st.subheader("⚠️ Đánh Giá Nhanh Model")

if len(df_model) < 30:
    st.warning("❌ Không đủ dữ liệu để đánh giá model")

else:
    try:
        # ===== MODEL =====
        X = df_model[["ma20","ma50","rsi"]]
        y = df_model["target"]

        model = train_model(df_model)
        preds = model.predict(X)

        # ===== CHECK 1 CHIỀU =====
        one_side_prediction = len(set(preds)) == 1

        # ===== METRICS =====
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        acc = accuracy_score(y, preds)
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)

        # ===== HIỂN THỊ =====
        if one_side_prediction:
            st.error("🚨 Model chỉ dự đoán 1 chiều → không đáng tin")

        if precision < 0.5:
            st.warning("⚠️ Tín hiệu mua yếu → dễ mua nhầm cổ phiếu xấu")

        if recall < 0.5:
            st.warning("⚠️ Dễ bỏ lỡ các nhịp tăng mạnh")

        if acc < 0.6:
            st.warning("⚠️ Cổ phiếu khó dự đoán → nên hạn chế trade")

        if f1 > 0.6:
            st.success("✅ Model tương đối ổn định")

    except Exception as e:
        st.error(f"Lỗi model: {e}")


st.subheader("📊 Phân Bố TARGET")

# Kiểm tra cột target tồn tại
if "target" not in df_model.columns:
    st.error("❌ Không tìm thấy cột 'target' trong dữ liệu")
else:
    # Tính tỷ lệ phân bố
    target_dist = df_model["target"].value_counts(normalize=True)

    up_pct = target_dist.get(1, 0)
    down_pct = target_dist.get(0, 0)

    # Hiển thị tỷ lệ %
    st.write(f"📈 Tỷ lệ TĂNG: {up_pct * 100:.1f}%")
    st.write(f"📉 Tỷ lệ GIẢM: {down_pct * 100:.1f}%")

    # Đánh giá mức độ lệch dữ liệu
    imbalance = abs(up_pct - down_pct)

    st.markdown("### 🧠 Nhận Định Dữ Liệu")

    if imbalance > 0.4:
        st.warning(
            "⚠️ Dữ liệu đang LỆCH RẤT MẠNH giữa hai lớp Tăng và Giảm.\n\n"
            "-> Điều này có thể khiến mô hình bị thiên lệch về lớp chiếm đa số,\n"
            "dẫn đến dự đoán kém chính xác cho lớp còn lại."
        )

    elif imbalance > 0.2:
        st.info(
            "📊 Dữ liệu có dấu hiệu lệch nhẹ.\n\n"
            "-> Mô hình có thể học tốt xu hướng chính, nhưng vẫn cần chú ý\n"
            "để tránh bỏ lỡ các tín hiệu quan trọng của lớp ít hơn."
        )

    else:
        st.success(
            "📊 Dữ liệu tương đối cân bằng.\n\n"
            "-> Đây là trạng thái tốt, giúp mô hình học ổn định hơn\n"
            "và không bị thiên lệch quá nhiều về một phía."
        )




st.write("---")
st.header("🤖 Hệ Thống Dự Báo và Độ Tin Cậy")

# 1. Tính toán Dự báo
features = ["ma20", "ma50", "rsi"]
latest = df_model.iloc[-1:]
pred = model.predict(latest[features])[0]
prob = model.predict_proba(latest[features])[0][1] if hasattr(model, "predict_proba") else 0.5

# Tính toán Sức mạnh Kỹ thuật (Technical Strength)
tech_score = 0
if latest['rsi'].values[0] < 45: tech_score += 0.25
if latest['Close'].values[0] > latest['ma20'].values[0]: tech_score += 0.25
combined_conf = (prob * 0.6) + tech_score  # AI chiếm 60%, Kỹ thuật 40%

# 2. Hiển thị Dashboard Độ tin cậy
c1, c2 = st.columns([1, 2])
with c1:
    color = "#00ffcc" if pred == 1 else "#ff4b4b"
    label = "GIÁ CỔ PHIẾU TĂNG" if pred == 1 else "GIÁ CỔ PHIẾU GIẢM"
    st.markdown(f"### Dự Báo Xu Hướng\n## <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

with c2:
    st.markdown(f"### 🎯 Độ Tin Cậy: {combined_conf*100:.1f}%")
    st.progress(min(combined_conf, 1.0))
    if combined_conf > 0.7: st.success("🔥 Tín hiệu Mạnh - Độ đồng thuận cao.")
    elif combined_conf > 0.5: st.info("📊 Tín hiệu Trung bình - Cần quan sát thêm.")
    else: st.warning("⚠️ Tín hiệu Rất Yếu - Thị trường đang nhiễu.")

# 3. Đánh giá Mô hình (Backtest)
with st.expander("📊 Chỉ số năng lực của Mô hình (Lịch sử)"):
    all_preds = model.predict(df_model[features])
    p_score = precision_score(y, all_preds, zero_division=0)
    f1 = f1_score(y, all_preds, zero_division=0)
    
    m1, m2 = st.columns(2)
    m1.metric("Độ chuẩn xác (Precision)", f"{p_score*100:.1f}%")
    m2.metric("Chỉ số F1 (Ổn định)", f"{f1:.2f}")

# 4. NHẬN ĐỊNH TỔNG HỢP (Đã fix mâu thuẫn)
st.write("---")
st.subheader("🧠 Nhận Định")

if combined_conf < 0.45:
    st.error(
        "🚨 **CẢNH BÁO: ĐỨNG NGOÀI QUAN SÁT**\n\n"
        "Mặc dù mô hình có lịch sử ổn định, nhưng tín hiệu hiện tại đang cực kỳ nhiễu "
        "và có độ tin cậy thấp" \
        "-> Tuyệt đối không nên mở vị thế giao dịch lúc này."
    )
elif p_score >= 0.7 and combined_conf >= 0.65:
    st.success(
        "✅ **TÍN HIỆU GIAO DỊCH TỐT**\n\n"
        "Mô hình hoạt động ổn định và có sự đồng thuận từ các chỉ báo kỹ thuật"
        "Dự báo có độ tin cậy cao" \
        "-> Phù hợp để cân nhắc giải ngân."
    )
elif p_score >= 0.55:
    st.info(
        "⚖️ **TÍN HIỆU THAM KHẢO**\n\n"
        "Mô hình có năng lực trung bình. Tín hiệu hiện tại không quá rõ ràng "
        "-> Chỉ nên giải ngân một phần nhỏ hoặc đợi thêm tín hiệu từ khối lượng giao dịch."
    )
else:
    st.warning(
        "🚫 **HẠN CHẾ GIAO DỊCH**\n\n"
        "Mô hình chưa cho thấy sự ổn định trong dữ liệu gần đây" \
        "-> Hãy kiểm tra lại các yếu tố tin tức vĩ mô trước khi quyết định."
    )


# =========================
# SIGNAL
# =========================
st.subheader("📍 Tín Hiệu Đề Xuất")

latest = df.iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric("Giá", round(latest["Close"],2))
col2.metric("Cắt lỗ", round(latest["stop_loss"],2))
col3.metric("Chốt lời", round(latest["take_profit"],2))

st.write(f"👉 Tín hiệu: {latest['signal']}")

# =========================
# DETAIL ANALYSIS
# =========================
# Gọi hàm khuyến nghị chi tiết
from features import get_detailed_advice, segment_customer

if st.session_state.data is not None:
    df_s = data[data["symbol"] == selected]
    advice_content = get_detailed_advice(df_s)
    current_seg = segment_customer(df_s)

    if advice_content:
        st.write("---")
        st.header(f"🎯 Chiến Lược Đầu Tư Mã {selected}")

        col_con, col_agg = st.columns(2)

        with col_con:
            st.success("🏠 **NHÓM THẬN TRỌNG (Conservative)**")
            res = advice_content["Conservative"]
            st.subheader(f"Hành động: {res['action']}")
            st.write(f"**Lý do:** {res['reason']}")
           

        with col_agg:
            st.warning("⚡ **NHÓM MẠO HIỂM (Aggressive)**")
            res = advice_content["Aggressive"]
            st.subheader(f"Hành động: {res['action']}")
            st.write(f"**Lý do:** {res['reason']}")
    
# =========================
# CAPITAL
# =========================
st.subheader("💰 Tối Ưu Nguồn Vốn")

von = st.number_input("Nhập vốn (VNĐ)", value=50000000)

qty = int(von // latest["Close"])

st.write(f"👉 Mua được: {qty:,} cổ phiếu {selected}")

# =========================
# BACKTEST
# =========================
from backtest import run_backtest

st.subheader("📈 Backtest")

df_bt, stats = run_backtest(df)

st.line_chart(df_bt.set_index("Date")[["equity"]])

st.write(f"💰 Lợi nhuận: {round(stats['total_return']*100,2)}%")
st.write(f"📉 Drawdown: {round(stats['max_drawdown'],2)}")
st.write(f"🔁 Số lệnh: {stats['num_trades']}")

# =========================
# =========================
# 🧠 PERFORMANCE SUMMARY (PRO LEVEL)
# =========================
st.subheader("🧠 Tổng Kết Hiệu Suất Chiến Lược")

profit = stats["total_return"]
drawdown = stats["max_drawdown"]
trades = stats["num_trades"]

# =========================
# 1. RETURN ANALYSIS
# =========================
if profit >= 0.20:
    profit_text = "Hiệu suất rất tốt – chiến lược có lợi thế rõ ràng"
    profit_level = "🔥 XUẤT SẮC"

elif profit >= 0.10:
    profit_text = "Hiệu suất tốt – có khả năng sinh lợi ổn định"
    profit_level = "📈 TỐT"

elif profit >= 0:
    profit_text = "Hiệu suất dương nhưng chưa ổn định"
    profit_level = "⚠️ TRUNG BÌNH"

else:
    profit_text = "Hiệu suất âm – chiến lược không hiệu quả"
    profit_level = "🚨 YẾU"


# =========================
# 2. RISK ANALYSIS (DRAWDOWN)
# =========================
if drawdown <= 0.05:
    risk_text = "Rủi ro thấp – biến động an toàn"
    risk_level = "🟢 AN TOÀN"

elif drawdown <= 0.15:
    risk_text = "Rủi ro trung bình – có biến động đáng kể"
    risk_level = "🟡 CHẤP NHẬN ĐƯỢC"

else:
    risk_text = "Rủi ro cao – dễ gây áp lực vốn"
    risk_level = "🔴 NGUY HIỂM"


# =========================
# 3. TRADE FREQUENCY
# =========================
if trades < 5:
    trade_text = "Chiến lược dài hạn – ít giao dịch"
elif trades < 15:
    trade_text = "Chiến lược swing – tần suất vừa phải"
else:
    trade_text = "Chiến lược active trading – giao dịch nhiều"


# =========================
# 4. DISPLAY DASHBOARD STYLE
# =========================
st.markdown("### 📊 Hiệu Suất")

st.write(f"📈 **Return:** {profit_level} | {profit*100:.2f}%")
st.write(f"📉 **Risk:** {risk_level} | Drawdown: {drawdown:.2%}")
st.write(f"🔁 **Trading Style:** {trade_text}")

# =========================
# 🧠 TỔNG KẾT (FIX LỖI LẶP)
# =========================
st.subheader("✨ Nhận Xét")

# ⚠️ CHỐT: đảm bảo chỉ render 1 lần duy nhất
summary_box = st.container()

with summary_box:

    # ===== HIỆU SUẤT =====
    if profit > 0.15:
        profit_text = "Hiệu suất rất tốt – Chiến lược sinh lợi mạnh"
        profit_icon = "🔥"
    elif profit > 0.08:
        profit_text = "Hiệu suất tốt – Có khả năng sinh lợi ổn định"
        profit_icon = "💰"
    elif profit > 0:
        profit_text = "Hiệu suất dương nhưng chưa ổn định"
        profit_icon = "⚠️"
    else:
        profit_text = "Hiệu suất âm – Chiến lược chưa hiệu quả"
        profit_icon = "🚨"

    # ===== RỦI RO =====
    if drawdown < 0.05:
        risk_text = "Rủi ro thấp – Khá an toàn"
        risk_icon = "🟢"
    elif drawdown < 0.15:
        risk_text = "Rủi ro trung bình – Cần cân nhắc"
        risk_icon = "🟡"
    else:
        risk_text = "Rủi ro cao – Dễ gây áp lực vốn"
        risk_icon = "📉"

    # ===== STYLE GIAO DỊCH =====
    if trades < 5:
        trade_text = "Đầu tư dài hạn – Ít giao dịch"
        trade_icon = "📊"
    elif trades < 15:
        trade_text = "Giao dịch vừa phải"
        trade_icon = "🔁"
    else:
        trade_text = "Giao dịch nhiều"
        trade_icon = "⚡"

    # =========================
    # 🔥 DISPLAY DUY NHẤT 1 LẦN
    # =========================
    st.markdown(f"{profit_icon} {profit_text}")

    st.markdown(f"{risk_icon} {risk_text}")

    st.markdown(f"{trade_icon} {trade_text}")

# =========================
# RISK
# =========================
st.subheader("📉 Phân Tích Rủi Ro")

returns = df_bt["Close"].pct_change().dropna()

vol = returns.std() * np.sqrt(252)

cum = (1 + returns).cumprod()
peak = cum.cummax()
dd = (cum - peak) / peak

max_dd = dd.min()

st.write(f" - Biến Động: {round(vol*100,2)}%")
st.write(f" - Mức Sụt Giảm Lớn Nhất: {round(max_dd*100,2)}%")

# =========================
# PORTFOLIO
# =========================
st.header("🔥 Top Cổ Phiếu VN30")

if "top_df" not in st.session_state:
    st.session_state.top_df = None

if st.button("🔍 Quét VN30"):

    with st.spinner("Đang phân tích thị trường..."):

        results = []

        symbols = data["symbol"].dropna().unique()

        for sym in symbols:

            df_s = data[data["symbol"] == sym].copy()

            df_s = df_s.dropna(subset=["ma20","ma50","rsi","Close"])

            if len(df_s) < 50:
                continue

            latest = df_s.iloc[-1]

            score = 0

            # ===== LOGIC CHẤM ĐIỂM (ỔN ĐỊNH) =====
            if latest["rsi"] < 40:
                score += 2   # vùng mua đẹp

            if latest["ma20"] > latest["ma50"]:
                score += 2   # xu hướng tăng

            if latest["Close"] > latest["ma20"]:
                score += 1   # giá khỏe

            if latest.get("signal","HOLD") == "BUY":
                score += 2

            if score >= 3:  # lọc rác
                results.append({
                    "symbol": sym,
                    "price": round(latest["Close"],2),
                    "rsi": round(latest["rsi"],1),
                    "trend": "Tăng" if latest["ma20"] > latest["ma50"] else "Yếu",
                    "score": score
                })

        if results:
            top_df = pd.DataFrame(results).sort_values("score", ascending=False)
            st.session_state.top_df = top_df.head(5)
        else:
            st.session_state.top_df = None

if st.session_state.top_df is not None:

    st.subheader("🏆 Top 5 MÃ CỔ PHIẾU ĐƯỢC ĐỀ XUẤT")

    st.dataframe(st.session_state.top_df, use_container_width=True)

    top1 = st.session_state.top_df.iloc[0]

    st.success(
        f"💡 Mã tốt nhất: {top1['symbol']} | Điểm: {top1['score']} | Giá: {top1['price']}"
    )

else:
    st.info("👉 Nhấn 'Quét VN30' Để Hiển Thị Top Mã Cổ Phiếu Được Đề Xuất!")

st.header("📊 Tối Ưu Nguồn Vốn")

# 👉 INPUT LUÔN HIỂN THỊ
von = st.number_input(
    "💰 Nhập Số Vốn Đầu Tư (VNĐ)",
    min_value=1_000_000,
    value=100_000_000,
    step=1_000_000
)
if st.button("📊 Tạo Danh Mục Đầu Tư"):

    if st.session_state.top_df is None:
        st.warning("👉 Hãy quét VN30 trước")
    else:

        df_port = st.session_state.top_df.copy()

        total_score = df_port["score"].sum()

        if total_score == 0:
            st.error("❌ Không đủ dữ liệu để phân bổ")
            st.stop()

        df_port["weight"] = df_port["score"] / total_score

        df_port["allocation"] = (df_port["weight"] * von).round(0)

        df_port["shares"] = (df_port["allocation"] / df_port["price"]).astype(int)

        st.subheader("🧾 Kết Luận Đầu Tư")

        st.write(f"Với số tiền {von:,.0f} VNĐ, bạn có thể mua:")

        for _, row in df_port.iterrows():
            st.write(f"- {row['shares']} cổ phiếu {row['symbol']}")
