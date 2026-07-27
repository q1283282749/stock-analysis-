import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="股票分析·腾讯数据", layout="wide")
st.title("📊 股票分析工具（腾讯数据 + 缠论 + 情绪）")

# ============ 数据获取（腾讯自选股） ============
def get_tencent_kline(symbol, days=500):
    """从腾讯财经获取前复权日K线，自动适配列数变化"""
    prefix = "sh" if symbol.startswith("6") else "sz"
    tc_code = prefix + symbol
    end_date = datetime.now().strftime("%Y-%m-%d")
    url = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{tc_code},day,,{end_date},{days},qfq"}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "http://stockapp.finance.qq.com/"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        node = data.get("data", {}).get(tc_code, {})
        klines = node.get("qfqday", node.get("day", []))
        if not klines:
            st.error("腾讯接口返回空数据，请检查股票代码")
            return None
        # 强制只取前6列（日期、开盘、收盘、最高、最低、成交量）
        klines = [row[:6] for row in klines]
        df = pd.DataFrame(klines, columns=["日期", "开盘", "收盘", "最高", "最低", "成交量"])
        df["成交量"] = pd.to_numeric(df["成交量"], errors="coerce") / 100
        df["日期"] = pd.to_datetime(df["日期"])
        for col in ["开盘", "收盘", "最高", "最低"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().sort_values("日期").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"腾讯数据获取失败: {e}")
        return None

# ============ 缠论分析 ============
def chanlun_analyze(df):
    """缠论：K线包含→分型→笔→中枢→买卖点"""
    df = df.copy()
    i = 1
    while i < len(df):
        ph, pl = df.loc[i-1, "最高"], df.loc[i-1, "最低"]
        ch, cl = df.loc[i, "最高"], df.loc[i, "最低"]
        if (ch >= ph and cl <= pl) or (ch <= ph and cl >= pl):
            if ch > ph:
                nh, nl = ch, pl
            else:
                nh, nl = ph, cl
            df.loc[i-1, "最高"] = nh
            df.loc[i-1, "最低"] = nl
            df = df.drop(index=i).reset_index(drop=True)
        else:
            i += 1

    tops, bottoms = [], []
    for i in range(1, len(df)-1):
        h1, l1 = df.loc[i-1, "最高"], df.loc[i-1, "最低"]
        h2, l2 = df.loc[i, "最高"], df.loc[i, "最低"]
        h3, l3 = df.loc[i+1, "最高"], df.loc[i+1, "最低"]
        if h2 > h1 and h2 > h3 and l2 > l1 and l2 > l3:
            tops.append(i)
        if l2 < l1 and l2 < l3 and h2 < h1 and h2 < h3:
            bottoms.append(i)

    bis = []
    if tops and bottoms:
        all_pts = sorted(tops + bottoms)
        prev = ("top", tops[0]) if tops[0] < bottoms[0] else ("bottom", bottoms[0])
        for p in all_pts[1:]:
            if (prev[0]=="top" and p in bottoms) or (prev[0]=="bottom" and p in tops):
                if abs(p - prev[1]) >= 2:
                    direction = 1 if prev[0]=="bottom" else -1
                    bis.append([prev[1], p, direction])
                    prev = ("top" if direction==1 else "bottom", p)

    zhongshus = []
    if len(bis) >= 3:
        for i in range(len(bis)-2):
            def get_range(idx):
                return (df.loc[idx, "最低"], df.loc[idx, "最高"])
            r1 = get_range(bis[i][0]); r1 = (min(r1[0], get_range(bis[i][1])[0]), max(r1[1], get_range(bis[i][1])[1]))
            r2 = get_range(bis[i+1][0]); r2 = (min(r2[0], get_range(bis[i+1][1])[0]), max(r2[1], get_range(bis[i+1][1])[1]))
            r3 = get_range(bis[i+2][0]); r3 = (min(r3[0], get_range(bis[i+2][1])[0]), max(r3[1], get_range(bis[i+2][1])[1]))
            ol = max(r1[0], r2[0], r3[0])
            oh = min(r1[1], r2[1], r3[1])
            if ol < oh:
                zhongshus.append({"top": oh, "bottom": ol})

    signal, confidence, reason = "观望", 0, ""
    if bis:
        last = bis[-1]
        last_dir, last_end = last[2], df.loc[last[1], "收盘"]
        if zhongshus:
            zs = zhongshus[-1]
            if last_dir == 1 and last_end > zs["top"]:
                signal, confidence, reason = "买入", 0.7, "向上笔突破中枢上沿，三买"
            elif last_dir == -1 and last_end < zs["bottom"]:
                signal, confidence, reason = "卖出", 0.7, "向下笔跌破中枢下沿，三卖"
            else:
                signal = "偏多" if last_dir == 1 else "偏空"
                confidence = 0.4
                reason = f"{'向上' if last_dir==1 else '向下'}笔未突破中枢"
        else:
            signal = "偏多" if last_dir == 1 else "偏空"
            confidence = 0.3
            reason = f"出现{'向上' if last_dir==1 else '向下'}笔"

    return {
        "signal": signal, "confidence": confidence, "reason": reason,
        "bi_count": len(bis), "zhongshu_count": len(zhongshus),
        "last_bi_dir": "向上" if (bis and bis[-1][2]==1) else "向下" if bis else "无",
        "last_price": df.loc[bis[-1][1], "收盘"] if bis else None
    }

# ============ 情绪周期 ============
def sentiment_score(params):
    lu = params.get("limit_up", 100)
    ld = params.get("limit_down", 10)
    bb = params.get("broken_board", 0.3)
    height = params.get("max_height", 3)
    score = min(lu/100,1)*25 + (1-ld/max(lu,1))*15 + (1-bb)*15 + min(height/7,1)*20 + min(params.get("turnover",8000)/10000,1)*15 + min(params.get("profit_effect",0.5),1)*10
    stage = "冰点" if score < 30 else "低迷" if score < 50 else "活跃" if score < 70 else "高潮"
    return {"score": round(score, 1), "stage": stage}

# ============ UI ============
with st.sidebar:
    st.header("⚙️ 参数设置")
    symbol = st.text_input("股票代码（如 000938）", "000938").strip()
    use_chan = st.checkbox("缠论分析", value=True)
    use_sentiment = st.checkbox("情绪周期", value=True)
    if use_sentiment:
        st.subheader("🎭 情绪参数")
        limit_up = st.slider("涨停家数", 0, 500, 100)
        limit_down = st.slider("跌停家数", 0, 500, 10)
        broken_board = st.slider("炸板率", 0.0, 1.0, 0.3)
        max_height = st.slider("最高连板", 1, 10, 3)
        profit_effect = st.slider("赚钱效应", 0.0, 1.0, 0.5)
        turnover = st.slider("成交额(亿)", 0, 20000, 8000)
    analyze_btn = st.button("🚀 执行分析", type="primary")

if analyze_btn:
    with st.spinner("正在获取腾讯数据并分析..."):
        df = get_tencent_kline(symbol)
        if df is None:
            st.stop()
        st.success(f"✅ 获取到 {len(df)} 条日K线")

        chan_res = None
        if use_chan:
            chan_res = chanlun_analyze(df)
            st.subheader("🐉 缠论分析")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("笔数", chan_res["bi_count"])
                st.metric("中枢数", chan_res["zhongshu_count"])
            with col2:
                st.metric("末笔方向", chan_res["last_bi_dir"])
                if chan_res["last_price"]:
                    st.metric("末笔收盘价", f"{chan_res['last_price']:.2f}")
            with col3:
                st.metric("缠论信号", chan_res["signal"])
                st.metric("置信度", f"{chan_res['confidence']*100:.0f}%")
            st.caption(chan_res["reason"])

        sent_res = None
        if use_sentiment:
            params = {
                "limit_up": limit_up, "limit_down": limit_down,
                "broken_board": broken_board, "max_height": max_height,
                "profit_effect": profit_effect, "turnover": turnover
            }
            sent_res = sentiment_score(params)
            st.subheader("🎭 情绪周期")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("综合得分", sent_res["score"])
            with col2:
                st.metric("当前阶段", sent_res["stage"])

        if chan_res and sent_res:
            st.markdown("---")
            st.subheader("🔗 综合信号")
            buy = 0; sell = 0
            if chan_res["signal"] == "买入": buy += 70
            elif chan_res["signal"] == "卖出": sell += 70
            elif "偏多" in chan_res["signal"]: buy += 20
            elif "偏空" in chan_res["signal"]: sell += 20
            if sent_res["stage"] == "高潮": sell += 30
            elif sent_res["stage"] == "冰点": buy += 30
            elif sent_res["stage"] == "低迷": buy += 15
            elif sent_res["stage"] == "活跃": buy += 10

            if buy > sell:
                final, conf = "买入", min(buy/100, 0.95)
            elif sell > buy:
                final, conf = "卖出", min(sell/100, 0.95)
            else:
                final, conf = "持有观望", 0.3

            color = {"买入": "green", "卖出": "red", "持有观望": "orange"}
            st.markdown(f"### 🎯 综合信号：<span style='color:{color[final]}'>{final}</span>", unsafe_allow_html=True)
            st.metric("置信度", f"{conf*100:.0f}%")

        with st.expander("查看最近K线"):
            st.dataframe(df.tail(30), use_container_width=True)
else:
    st.info("👈 请输入股票代码并点击「执行分析」")
