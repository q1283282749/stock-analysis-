import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import math

st.set_page_config(page_title="股票综合分析工具", layout="wide")
st.title("📊 股票综合分析工具（可选指标）")
st.markdown("数据来源：akshare（免费A股实时数据） | 所有计算在本地完成")

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ 参数设置")
    symbol = st.text_input("股票代码（如 000938）", value="000938").strip()
    st.markdown("---")
    st.subheader("📈 技术指标")
    use_sma = st.checkbox("SMA(20)", value=True)
    use_ema = st.checkbox("EMA(20)", value=True)
    use_rsi = st.checkbox("RSI(14)", value=True)
    use_macd = st.checkbox("MACD(12,26,9)", value=True)
    use_boll = st.checkbox("布林带(20,2)", value=True)
    use_kdj = st.checkbox("KDJ(9,3,3)", value=True)
    use_atr = st.checkbox("ATR(14)", value=True)
    use_wr = st.checkbox("威廉%R(14)", value=True)
    use_obv = st.checkbox("OBV", value=True)
    st.markdown("---")
    st.subheader("🐉 缠论结构")
    use_chan = st.checkbox("启用缠论分析", value=True)
    st.markdown("---")
    st.subheader("🎭 情绪周期")
    use_sentiment = st.checkbox("启用情绪周期", value=True)
    if use_sentiment:
        col1, col2 = st.columns(2)
        with col1:
            limit_up = st.number_input("涨停家数", value=85)
            limit_down = st.number_input("跌停家数", value=5)
            broken_board = st.number_input("炸板家数", value=18)
            max_height = st.number_input("最高连板高度", value=7)
        with col2:
            profit_effect = st.number_input("昨日涨停平均涨幅(%)", value=3.2)
            turnover = st.number_input("今日成交额(亿)", value=152.0)
            turnover_ma = st.number_input("近20日均成交额(亿)", value=96.0)
            advancing = st.number_input("上涨家数", value=2100)
            declining = st.number_input("下跌家数", value=1850)
    st.markdown("---")
    st.subheader("🔗 综合信号")
    use_combined = st.checkbox("启用综合信号（需缠论+情绪）", value=True)
    analyze_btn = st.button("🚀 执行分析", type="primary")

# ---------- 核心算法 ----------
def calc_sma(prices, period):
    result = []
    for i in range(period - 1, len(prices)):
        result.append(np.mean(prices[i-period+1:i+1]))
    return result

def calc_ema(prices, period):
    if len(prices) < period:
        return []
    multiplier = 2 / (period + 1)
    ema = np.mean(prices[:period])
    result = [ema]
    for p in prices[period:]:
        ema = (p - ema) * multiplier + ema
        result.append(ema)
    return result

def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return []
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi = []
    if avg_loss == 0:
        rsi.append(100)
    else:
        rsi.append(100 - 100 / (1 + avg_gain/avg_loss))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi.append(100)
        else:
            rsi.append(100 - 100 / (1 + avg_gain/avg_loss))
    return rsi

def calc_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(prices, fast)
    ema_slow = calc_ema(prices, slow)
    if len(ema_fast) != len(ema_slow):
        return None
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = calc_ema(macd_line, signal)
    min_len = min(len(macd_line), len(signal_line))
    histogram = [macd_line[-min_len + i] - signal_line[i] for i in range(min_len)]
    return {'macdLine': macd_line[-min_len:], 'signalLine': signal_line, 'histogram': histogram}

def calc_bollinger(prices, period=20, multiplier=2):
    sma = calc_sma(prices, period)
    upper, lower = [], []
    for i, m in enumerate(sma):
        start = i + period - 1
        subset = prices[start-period+1:start+1]
        std = np.std(subset)
        upper.append(m + multiplier * std)
        lower.append(m - multiplier * std)
    return {'upper': upper, 'middle': sma, 'lower': lower}

def calc_kdj(high, low, close, period=9, k_factor=1/3, d_factor=1/3):
    if len(close) < period:
        return None
    rsv = []
    for i in range(period-1, len(close)):
        hh = max(high[i-period+1:i+1])
        ll = min(low[i-period+1:i+1])
        if hh == ll:
            rsv.append(50)
        else:
            rsv.append((close[i] - ll) / (hh - ll) * 100)
    K, D = 50, 50
    k_arr, d_arr, j_arr = [], [], []
    for v in rsv:
        K = k_factor * v + (1 - k_factor) * K
        D = d_factor * K + (1 - d_factor) * D
        J = 3 * K - 2 * D
        k_arr.append(K)
        d_arr.append(D)
        j_arr.append(J)
    return {'K': k_arr, 'D': d_arr, 'J': j_arr}

def calc_atr(high, low, close, period=14):
    if len(high) < 2:
        return []
    tr = []
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr.append(max(hl, hc, lc))
    return calc_ema(tr, period)

def calc_williams_r(high, low, close, period=14):
    if len(close) < period:
        return []
    result = []
    for i in range(period-1, len(close)):
        hh = max(high[i-period+1:i+1])
        ll = min(low[i-period+1:i+1])
        if hh == ll:
            result.append(-50)
        else:
            result.append((hh - close[i]) / (hh - ll) * -100)
    return result

def calc_obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i-1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])
    return obv

# 缠论
def process_containment(klines):
    if len(klines) == 0:
        return []
    processed = [klines[0]]
    for cur in klines[1:]:
        last = processed[-1]
        up_dir = (last['high'] < cur['high'] and last['low'] < cur['low'])
        down_dir = (last['high'] > cur['high'] and last['low'] > cur['low'])
        contain = (cur['high'] <= last['high'] and cur['low'] >= last['low']) or \
                  (cur['high'] >= last['high'] and cur['low'] <= last['low'])
        if contain:
            if up_dir:
                new_high = max(last['high'], cur['high'])
                new_low = max(last['low'], cur['low'])
            elif down_dir:
                new_high = min(last['high'], cur['high'])
                new_low = min(last['low'], cur['low'])
            else:
                new_high = max(last['high'], cur['high'])
                new_low = min(last['low'], cur['low'])
            processed[-1] = {'date': last['date'], 'open': last['open'], 'close': cur['close'],
                             'high': new_high, 'low': new_low, 'volume': last['volume'] + cur['volume']}
        else:
            processed.append(cur)
    return processed

def find_fractals(processed):
    if len(processed) < 3:
        return []
    fractals = []
    for i in range(1, len(processed)-1):
        left, mid, right = processed[i-1], processed[i], processed[i+1]
        is_top = (mid['high'] > left['high'] and mid['high'] > right['high'] and
                  mid['low'] > left['low'] and mid['low'] > right['low'])
        is_bottom = (mid['low'] < left['low'] and mid['low'] < right['low'] and
                     mid['high'] < left['high'] and mid['high'] < right['high'])
        if is_top:
            fractals.append({'isTop': True, 'index': i, 'price': mid['high']})
        elif is_bottom:
            fractals.append({'isTop': False, 'index': i, 'price': mid['low']})
    return fractals

def build_bis(fractals):
    if len(fractals) < 2:
        return []
    bis = []
    prev = None
    for f in fractals:
        if prev is None:
            prev = f
            continue
        if prev['isTop'] == f['isTop']:
            if f['isTop'] and f['price'] > prev['price']:
                prev = f
            elif not f['isTop'] and f['price'] < prev['price']:
                prev = f
            continue
        if abs(f['index'] - prev['index']) < 2:
            prev = f
            continue
        is_up = not prev['isTop']
        bis.append({'startIndex': prev['index'], 'endIndex': f['index'], 'isUp': is_up,
                    'startPrice': prev['price'], 'endPrice': f['price']})
        prev = f
    return bis

def find_zhongshus(bis):
    if len(bis) < 3:
        return []
    zhongshus = []
    for i in range(len(bis)-2):
        b1, b2, b3 = bis[i], bis[i+1], bis[i+2]
        r1_low = min(b1['startPrice'], b1['endPrice'])
        r1_high = max(b1['startPrice'], b1['endPrice'])
        r2_low = min(b2['startPrice'], b2['endPrice'])
        r2_high = max(b2['startPrice'], b2['endPrice'])
        r3_low = min(b3['startPrice'], b3['endPrice'])
        r3_high = max(b3['startPrice'], b3['endPrice'])
        overlap_low = max(r1_low, r2_low, r3_low)
        overlap_high = min(r1_high, r2_high, r3_high)
        if overlap_low < overlap_high:
            zhongshus.append({'top': overlap_high, 'bottom': overlap_low, 'startIndex': i, 'endIndex': i+2})
    return zhongshus

# 情绪周期
def evaluate_sentiment(params):
    lu = params['limit_up']; ld = params['limit_down']; bb = params['broken_board']
    height = params['max_height']; yz = params['profit_effect'] / 100.0
    today = params['turnover'] * 1e8; ma20 = params['turnover_ma'] * 1e8
    adv = params['advancing']; dec = params['declining']
    lu_score = 0
    if lu >= 120: lu_score = 20
    elif lu >= 60: lu_score = 10 + (lu - 60) * (10/60)
    elif lu >= 30: lu_score = 5 + (lu - 30) * (5/30)
    else: lu_score = lu * (5/30)
    bh_score = 0
    if height >= 10: bh_score = 20
    elif height >= 7: bh_score = 15 + (height - 7) * (5/3)
    elif height >= 5: bh_score = 10 + (height - 5) * (5/2)
    elif height >= 3: bh_score = 5 + (height - 3) * (5/2)
    else: bh_score = height * (5/3)
    total_attempt = lu + bb
    broken_rate = bb / total_attempt if total_attempt > 0 else 1
    br_score = 0
    if broken_rate <= 0.1: br_score = 15
    elif broken_rate <= 0.2: br_score = 15 - (broken_rate - 0.1) * 50
    elif broken_rate <= 0.35: br_score = 10 - (broken_rate - 0.2) * (5/0.15)
    elif broken_rate <= 0.5: br_score = 5 - (broken_rate - 0.35) * (5/0.15)
    else: br_score = 0
    br_score = max(0, br_score)
    pe_score = 0
    if yz >= 0.05: pe_score = 15
    elif yz >= 0.03: pe_score = 10 + (yz - 0.03) * 250
    elif yz >= 0: pe_score = 5 + yz * (5/0.03)
    elif yz >= -0.03: pe_score = (yz + 0.03) * (5/0.03)
    else: pe_score = 0
    pe_score = max(0, min(15, pe_score))
    ratio = today / ma20 if ma20 > 0 else 1
    to_score = 0
    if ratio >= 1.5: to_score = 15
    elif ratio >= 1.2: to_score = 10 + (ratio - 1.2) * (5/0.3)
    elif ratio >= 1.0: to_score = 5 + (ratio - 1.0) * (5/0.2)
    elif ratio >= 0.8: to_score = (ratio - 0.8) * (5/0.2)
    else: to_score = 0
    to_score = max(0, min(15, to_score))
    ld_ratio = ld / lu if lu > 0 else 1
    ld_score = 0
    if ld_ratio <= 0.1: ld_score = 8
    elif ld_ratio <= 0.3: ld_score = 8 - (ld_ratio - 0.1) * 15
    elif ld_ratio <= 0.6: ld_score = 5 - (ld_ratio - 0.3) * (3/0.3)
    elif ld_ratio <= 1.0: ld_score = 2 - (ld_ratio - 0.6) * (2/0.4)
    else: ld_score = 0
    ld_score = max(0, ld_score)
    width_ratio = adv / dec if dec > 0 else (7 if adv > 0 else 0)
    w_score = 0
    if width_ratio >= 2.0: w_score = 7
    elif width_ratio >= 1.0: w_score = 4 + (width_ratio - 1.0) * 3
    elif width_ratio >= 0.5: w_score = 1 + (width_ratio - 0.5) * 6
    else: w_score = width_ratio * 2
    w_score = max(0, min(7, w_score))
    total = lu_score + bh_score + br_score + pe_score + to_score + ld_score + w_score
    total = min(100, total)
    if total < 15: stage = '冰点期'
    elif total < 35: stage = '低温期'
    elif total < 60: stage = '回暖期'
    elif total < 80: stage = '高潮期'
    else: stage = '过热期'
    return {'total': total, 'stage': stage, 'breakdown': {'luScore': lu_score, 'bhScore': bh_score,
            'brScore': br_score, 'peScore': pe_score, 'toScore': to_score, 'ldScore': ld_score, 'wScore': w_score}}

def combined_signal(sentiment, bis, zhongshus):
    if len(bis) == 0:
        return {'signal': '无信号', 'confidence': 0, 'reason': '缺乏缠论笔结构'}
    last_bi = bis[-1]
    last_is_up = last_bi['isUp']
    last_end_price = last_bi['endPrice']
    nearest_zs = zhongshus[-1] if zhongshus else None
    break_up = break_down = False
    if nearest_zs:
        if last_is_up and last_end_price > nearest_zs['top']:
            break_up = True
        elif not last_is_up and last_end_price < nearest_zs['bottom']:
            break_down = True
    divergence_bullish = divergence_bearish = False
    if len(bis) >= 2:
        second_last = bis[-2]
        last_len = abs(last_bi['endPrice'] - last_bi['startPrice'])
        second_len = abs(second_last['endPrice'] - second_last['startPrice'])
        if not last_is_up and not second_last['isUp'] and last_end_price < second_last['endPrice'] and last_len < second_len * 0.7:
            divergence_bullish = True
        if last_is_up and second_last['isUp'] and last_end_price > second_last['endPrice'] and last_len < second_len * 0.7:
            divergence_bearish = True
    stage = sentiment['stage']
    if stage == '冰点期' and divergence_bullish:
        return {'signal': '强买入', 'confidence': 0.85, 'reason': '冰点期出现笔底背离，情绪冰点+结构背离双重共振，强烈买入信号'}
    if stage == '低温期' and not last_is_up and nearest_zs and last_end_price >= nearest_zs['bottom']:
        return {'signal': '弱买入', 'confidence': 0.65, 'reason': '低温期价格回落至中枢下沿，存在支撑，可轻仓试多'}
    if stage == '回暖期' and break_up:
        return {'signal': '弱买入', 'confidence': 0.70, 'reason': '回暖期中枢向上突破，情绪回暖配合技术突破，买入'}
    if stage == '高潮期' and divergence_bearish:
        return {'signal': '弱卖出', 'confidence': 0.75, 'reason': '高潮期出现顶背离，情绪过热叠加结构背离，减仓'}
    if stage == '过热期' and last_is_up and nearest_zs and last_end_price > nearest_zs['top'] * 1.1:
        return {'signal': '强卖出', 'confidence': 0.90, 'reason': '过热期价格远离中枢上沿，情绪极端亢奋，强烈卖出'}
    if nearest_zs and not break_up and not break_down:
        if stage in ('冰点期', '低温期'):
            return {'signal': '持有观望', 'confidence': 0.40, 'reason': '情绪低迷，中枢内震荡，等待方向选择'}
        if stage in ('高潮期', '过热期'):
            return {'signal': '弱卖出', 'confidence': 0.55, 'reason': '情绪偏高，中枢内不宜追高，警惕回调'}
        return {'signal': '持有观望', 'confidence': 0.45, 'reason': '情绪中性，中枢内观望'}
    return {'signal': '持有观望', 'confidence': 0.30, 'reason': '无明显共振信号，建议观望'}

# ---------- 数据获取 ----------
@st.cache_data(ttl=300)
def fetch_data(symbol):
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
        if df.empty:
            st.error(f"未找到股票 {symbol} 的数据，请检查代码")
            return None
        df = df.sort_values("日期")
        klines = []
        for _, row in df.iterrows():
            klines.append({
                'date': str(row['日期']),
                'open': float(row['开盘']),
                'close': float(row['收盘']),
                'high': float(row['最高']),
                'low': float(row['最低']),
                'volume': float(row['成交量'])
            })
        return klines
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None

# ---------- 主逻辑 ----------
if analyze_btn:
    with st.spinner("正在获取数据并分析..."):
        klines = fetch_data(symbol)
        if klines is None:
            st.stop()
        closes = [k['close'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        volumes = [k['volume'] for k in klines]

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("📈 技术指标最新值")
            tech_lines = []
            if use_sma:
                sma = calc_sma(closes, 20)
                tech_lines.append(f"SMA(20): {sma[-1]:.2f}" if sma else "SMA: N/A")
            if use_ema:
                ema = calc_ema(closes, 20)
                tech_lines.append(f"EMA(20): {ema[-1]:.2f}" if ema else "EMA: N/A")
            if use_rsi:
                rsi = calc_rsi(closes, 14)
                tech_lines.append(f"RSI(14): {rsi[-1]:.2f}" if rsi else "RSI: N/A")
            if use_macd:
                macd = calc_macd(closes, 12, 26, 9)
                if macd:
                    tech_lines.append(f"MACD红柱: {macd['histogram'][-1]:.4f}")
            if use_boll:
                boll = calc_bollinger(closes, 20, 2)
                if boll:
                    tech_lines.append(f"布林上轨: {boll['upper'][-1]:.2f} | 中轨: {boll['middle'][-1]:.2f} | 下轨: {boll['lower'][-1]:.2f}")
            if use_kdj:
                kdj = calc_kdj(highs, lows, closes, 9, 1/3, 1/3)
                if kdj:
                    tech_lines.append(f"KDJ: K={kdj['K'][-1]:.2f} D={kdj['D'][-1]:.2f} J={kdj['J'][-1]:.2f}")
            if use_atr:
                atr = calc_atr(highs, lows, closes, 14)
                tech_lines.append(f"ATR(14): {atr[-1]:.4f}" if atr else "ATR: N/A")
            if use_wr:
                wr = calc_williams_r(highs, lows, closes, 14)
                tech_lines.append(f"威廉%R(14): {wr[-1]:.2f}" if wr else "WR: N/A")
            if use_obv:
                obv = calc_obv(closes, volumes)
                tech_lines.append(f"OBV: {obv[-1]:.0f}" if obv else "OBV: N/A")
            for line in tech_lines:
                st.write(line)

        with col_right:
            if use_chan:
                st.subheader("🐉 缠论结构")
                processed = process_containment(klines)
                fractals = find_fractals(processed)
                bis = build_bis(fractals)
                zhongshus = find_zhongshus(bis)
                st.write(f"包含处理后K线数: {len(processed)}")
                st.write(f"分型数: {len(fractals)}")
                st.write(f"笔数: {len(bis)}")
                st.write(f"中枢数: {len(zhongshus)}")
                if bis:
                    last = bis[-1]
                    st.write(f"最后一笔方向: {'向上' if last['isUp'] else '向下'}")
                    st.write(f"区间: {last['startPrice']:.2f} → {last['endPrice']:.2f}")
                if zhongshus:
                    zs = zhongshus[-1]
                    st.write(f"最近中枢: 上沿 {zs['top']:.2f}, 下沿 {zs['bottom']:.2f}")

        if use_sentiment:
            st.subheader("🎭 情绪周期")
            params = {
                'limit_up': limit_up, 'limit_down': limit_down, 'broken_board': broken_board,
                'max_height': max_height, 'profit_effect': profit_effect, 'turnover': turnover,
                'turnover_ma': turnover_ma, 'advancing': advancing, 'declining': declining
            }
            sentiment = evaluate_sentiment(params)
            st.write(f"综合得分: {sentiment['total']:.1f} 分")
            st.write(f"当前阶段: {sentiment['stage']}")
            st.write("细分得分:")
            bd = sentiment['breakdown']
            st.write(f"  涨停家数: {bd['luScore']:.1f}/20")
            st.write(f"  最高连板: {bd['bhScore']:.1f}/20")
            st.write(f"  炸板率: {bd['brScore']:.1f}/15")
            st.write(f"  赚钱效应: {bd['peScore']:.1f}/15")
            st.write(f"  成交额: {bd['toScore']:.1f}/15")
            st.write(f"  跌停对比: {bd['ldScore']:.1f}/8")
            st.write(f"  市场宽度: {bd['wScore']:.1f}/7")

        if use_combined and use_chan and use_sentiment:
            st.subheader("🔗 综合信号")
            signal = combined_signal(sentiment, bis, zhongshus)
            st.write(f"信号: {signal['signal']}")
            st.write(f"置信度: {signal['confidence']*100:.0f}%")
            st.write(f"理由: {signal['reason']}")

        with st.expander("查看最近K线数据"):
            df_display = pd.DataFrame(klines[-30:])
            st.dataframe(df_display, use_container_width=True)

else:
    st.info("👈 请在侧边栏配置参数后点击「执行分析」")
    st.markdown("""
    ### 使用说明
    1. 输入股票代码（如 000938 代表紫光股份）
    2. 勾选你想要使用的指标
    3. 情绪周期参数可按实际情况填写（可先用默认值测试）
    4. 点击「执行分析」查看结果
    """)
