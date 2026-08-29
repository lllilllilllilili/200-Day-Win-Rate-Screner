"""
200일선 위치별 승률 스크리너
================================
종목 검색 → 200일선 대비 위치를 구간별로 쪼개서
거래수 / 승률 / 평균수익 / 최대수익 / 평균보유일수 를 전수조사 테이블로 보여준다.
현재 위치에 해당하는 행을 하이라이트해서 "지금 여기서 사면 역사적으로 어땠는지"를 직감적으로 확인.

데이터: Yahoo Finance (미국주식/ETF, 코인),
        한국주식은 FinanceDataReader fallback
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="200일선 승률 스크리너", page_icon="📈", layout="wide")

# -- Custom CSS for dark table styling --
st.markdown("""
<style>
.position-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Pretendard', -apple-system, sans-serif;
    font-size: 14px;
    margin-top: 16px;
}
.position-table th {
    text-align: left;
    padding: 8px 12px;
    border-bottom: 2px solid #444;
    color: #aaa;
    font-weight: 600;
}
.position-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #333;
}
.position-table tr.current-row {
    background: rgba(59, 130, 246, 0.2);
    border-left: 3px solid #3b82f6;
}
.position-table tr.current-row td {
    font-weight: 700;
    color: #60a5fa;
}
.win-high { color: #34d399; font-weight: 600; }
.win-mid { color: #fbbf24; }
.win-low { color: #f87171; }
.metric-box {
    border: 1px solid #444;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-box .value { font-size: 28px; font-weight: 700; }
.metric-box .label { font-size: 12px; color: #999; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 데이터 로딩
# ------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def load_krx_listing() -> pd.DataFrame:
    """한국 상장 종목 전체 목록(코드+이름). 실패 시 빈 DF."""
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KRX")
        return df[["Code", "Name", "Market"]].copy()
    except Exception:
        return pd.DataFrame(columns=["Code", "Name", "Market"])


def _has_korean(text: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)


# 영문 약자 → 한글 발음 별칭.
# 상장명에 영문 약자가 섞여 있어(예: 'NAVER', 'SK하이닉스') 한글 발음으로
# 검색할 때 매칭되도록, 검색어의 한글 발음을 영문 약자로 되돌려 준다.
_ALIAS_KO2EN = {
    # 단독 브랜드
    "네이버": "NAVER",
    "포스코": "POSCO", "에이치엠엠": "HMM",
    # 영문 약자 접두 (긴 것부터 매칭되도록 dict 순서는 무관 - 코드에서 처리)
    "에스케이": "SK", "에스k": "SK",
    "엘지": "LG", "엘쥐": "LG",
    "케이티앤지": "KT&G", "케이티지": "KT&G", "케이티": "KT", "케이비": "KB",
    "엔에이치": "NH", "엔에이치엔": "NHN",
    "엘에스": "LS", "엘엑스": "LX", "지에스": "GS", "씨제이": "CJ",
    "제이비": "JB", "제이더블유": "JW", "디비": "DB", "디엘": "DL",
    "케이지": "KG", "에이치디": "HD", "에이치엘비": "HLB", "에이치엘": "HL",
    "에이치디씨": "HDC", "에스티엑스": "STX", "나이스": "NICE", "에스에프에이": "SFA",
    "케이씨씨": "KCC", "비지에프": "BGF", "에스지": "SG", "에스엔티": "SNT",
    "에이치비": "HB", "에이치에스": "HS", "디에이치": "DH",
    # 흔한 개별 종목
    "에이피알": "APR", "에코프로에이치엔": "에코프로HN",
    "에스디바이오센서": "SD바이오센서",
}


def _looks_like_ticker(q: str) -> bool:
    """
    yfinance 티커처럼 생겼는지 (한국종목 검색을 건너뛸지 판단).
    예: AAPL, TQQQ, BTC-USD, ^GSPC, BRK.B, 005930.KS
    영문 대문자/숫자와 . - ^ = 로만 구성되면 티커로 본다.
    단, 순수 알파벳 소문자 단어(naver 등)는 티커로 보지 않아 한국종목 검색을 태운다.
    """
    import re
    s = q.strip()
    if not s:
        return False
    # 숫자코드(005930) 또는 .KS/.KQ 접미사는 티커로 취급
    if re.fullmatch(r"\d{6}(\.[A-Za-z]{2})?", s):
        return True
    # 지수 심볼 (^GSPC, ^IXIC 등)
    if s.startswith("^"):
        return True
    # 대문자/숫자 + 특수문자 조합 (소문자 없음)
    if re.fullmatch(r"[A-Z0-9][A-Z0-9.\-^=]*", s):
        return True
    return False


def _apply_alias(q: str) -> str:
    """검색어 전체 또는 접두어가 별칭이면 영문 약자로 치환. 긴 별칭 우선."""
    low = q.replace(" ", "").lower()
    # 긴 별칭부터 검사해서 '케이티앤지'가 '케이티'보다 먼저 매칭되게 함
    for ko in sorted(_ALIAS_KO2EN, key=len, reverse=True):
        en = _ALIAS_KO2EN[ko]
        ko_low = ko.lower()
        if low == ko_low:
            return en
        if low.startswith(ko_low):
            return en + q.replace(" ", "")[len(ko):]  # '에스케이하이닉스' -> 'SK하이닉스'
    return q


def resolve_korean_name(query: str):
    """
    기업명(한글/영문/별칭)을 종목코드로 변환. 대소문자 무시.
    Returns:
        ("code", "005930", "삼성전자")  정확/단독 매칭 성공
        ("candidates", [(code, name, market), ...])  후보 여러 개
        ("none", [])  매칭 없음
    """
    listing = load_krx_listing()
    if listing.empty:
        return ("none", [])

    raw_q = query.strip()
    names = listing["Name"].astype(str)
    names_low = names.str.lower()

    # 후보 검색어들: 별칭 치환본을 우선 시도 (더 정확한 의도), 그 다음 원본
    aliased = _apply_alias(raw_q).strip()
    queries = []
    for cand in (aliased, raw_q):
        c = cand.strip()
        if c and c not in queries:
            queries.append(c)

    for q in queries:
        q_low = q.lower()

        # 1) 완전 일치 (대소문자 무시)
        exact = listing[names_low == q_low]
        if len(exact) == 1:
            r = exact.iloc[0]
            return ("code", str(r["Code"]), str(r["Name"]))
        if len(exact) > 1:
            return ("candidates", [(str(r["Code"]), str(r["Name"]), str(r["Market"]))
                                   for _, r in exact.iterrows()])

        # 2) 부분 일치 (앞부분 우선, 대소문자 무시)
        starts = listing[names_low.str.startswith(q_low)]
        contains = listing[names_low.str.contains(q_low, regex=False)]
        merged = pd.concat([starts, contains]).drop_duplicates(subset="Code")

        if len(merged) == 1:
            r = merged.iloc[0]
            return ("code", str(r["Code"]), str(r["Name"]))
        if len(merged) > 1:
            cands = [(str(r["Code"]), str(r["Name"]), str(r["Market"]))
                     for _, r in merged.head(10).iterrows()]
            return ("candidates", cands)

    return ("none", [])


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str) -> pd.DataFrame:
    ticker = ticker.strip()
    df = _load_yfinance(ticker)
    if df is not None and len(df) >= 250:
        return df
    df_fdr = _load_fdr(ticker)
    if df_fdr is not None and len(df_fdr) >= 250:
        return df_fdr
    if df is not None and len(df) > 0:
        return df
    if df_fdr is not None and len(df_fdr) > 0:
        return df_fdr
    return pd.DataFrame()


def _load_yfinance(ticker: str):
    try:
        raw = yf.download(ticker, start="2000-01-01", auto_adjust=True, progress=False)
        if raw is None or len(raw) == 0:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if "Close" not in raw.columns:
            return None
        df = raw[["Close"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().dropna()
        return df
    except Exception:
        return None


def _load_fdr(ticker: str):
    try:
        import FinanceDataReader as fdr
    except Exception:
        return None
    code = ticker.replace(".KS", "").replace(".KQ", "")
    try:
        raw = fdr.DataReader(code, "2000-01-01")
        if raw is None or len(raw) == 0 or "Close" not in raw.columns:
            return None
        df = raw[["Close"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().dropna()
        return df
    except Exception:
        return None


# ------------------------------------------------------------
# 핵심 분석 로직
# ------------------------------------------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA200"] = out["Close"].rolling(200).mean()
    out = out.dropna()
    out["gap"] = (out["Close"] - out["SMA200"]) / out["SMA200"] * 100
    return out


def zone_analysis(df: pd.DataFrame, band_width: float, step: float,
                  target_pct: float, stop_pct: float, max_hold: int,
                  zone_min: float = -50, zone_max: float = 50) -> pd.DataFrame:
    """
    200일선 괴리율을 슬라이딩 구간으로 나누고,
    각 구간에서 매수 후 "목표수익 도달 vs 손절 중 먼저 닿는 것"으로 성과를 전수조사.

    구간 생성 (슬라이딩 윈도우):
      - 중심(center)을 0%부터 step% 간격으로 양/음 방향으로 찍는다.
      - 각 중심마다 [center - band_width/2, center + band_width/2] 를 구간으로.
      - step < band_width 이면 인접 구간이 겹친다 (한 날이 여러 구간에 중복 집계).
      - step == band_width 이면 겹치지 않는다.

    판정:
      - 목표수익(target_pct%)에 먼저 닿으면 승리(익절)
      - 손절(stop_pct%)에 먼저 닿으면 패배
      - max_hold 거래일 안에 둘 다 안 닿으면, 만기 시점 수익률 부호로 판정

    Returns DataFrame with columns:
        center, zone_label, trades, win_rate, avg_return,
        max_return, avg_holding_days
    """
    close = df["Close"].values
    gap = df["gap"].values
    n = len(close)
    half = band_width / 2

    # 0%를 중심으로 step 간격의 중심점들 생성 (대칭)
    n_neg = int(np.floor((0 - zone_min) / step))
    n_pos = int(np.floor((zone_max - 0) / step))
    centers = [round(-k * step, 6) for k in range(n_neg, 0, -1)] + \
              [round(k * step, 6) for k in range(0, n_pos + 1)]

    rows = []

    def fmt(v):
        return f"{v:+.0f}" if abs(v - round(v)) < 1e-9 else f"{v:+.1f}"

    for center in centers:
        lo, hi = center - half, center + half
        zone_label = f"{fmt(lo)}%~{fmt(hi)}%"
        mask = (gap >= lo) & (gap < hi)
        positions = np.where(mask)[0]

        exit_returns = []   # 청산 시 실제 수익률(%)
        max_rets = []       # 보유 중 최대 도달 수익률(%)
        hold_days = []      # 청산까지 걸린 거래일수
        wins = 0

        for pos in positions:
            if pos >= n - 1:
                continue
            entry = close[pos]
            end = min(pos + max_hold, n - 1)
            path = close[pos + 1:end + 1]
            if len(path) == 0:
                continue

            cum = (path / entry - 1) * 100  # 진입 이후 일별 수익률
            max_rets.append(float(cum.max()))

            hit_target = np.where(cum >= target_pct)[0]
            hit_stop = np.where(cum <= -stop_pct)[0]
            t_day = hit_target[0] if len(hit_target) else None
            s_day = hit_stop[0] if len(hit_stop) else None

            if t_day is not None and (s_day is None or t_day <= s_day):
                # 목표 먼저 (동시 캔들이면 목표 우선 처리)
                wins += 1
                exit_returns.append(target_pct)
                hold_days.append(int(t_day) + 1)
            elif s_day is not None:
                # 손절 먼저
                exit_returns.append(-stop_pct)
                hold_days.append(int(s_day) + 1)
            else:
                # 만기까지 미달 -> 만기 수익률 부호로 판정
                final = float(cum[-1])
                if final > 0:
                    wins += 1
                exit_returns.append(final)
                hold_days.append(len(path))

        trades = len(exit_returns)
        if trades == 0:
            continue

        exit_returns = np.array(exit_returns)
        max_rets = np.array(max_rets)
        hold_days = np.array(hold_days)

        rows.append({
            "center": center,
            "zone_label": zone_label,
            "trades": trades,
            "win_rate": float(wins / trades * 100),
            "avg_return": float(exit_returns.mean()),
            "max_return": float(max_rets.mean()),
            "avg_holding_days": int(round(hold_days.mean())),
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("📈 200일선 위치별 승률 스크리너")
st.caption("종목 검색 → 200일선 대비 모든 위치 구간의 역사적 승률을 한눈에")

with st.expander("💡 사용법 & 티커 예시", expanded=False):
    st.markdown("""
- **미국주식/ETF**: `AAPL`, `TSLA`, `QQQ`, `TQQQ`, `SOXL`
- **코인**: `BTC-USD`, `ETH-USD`, `SOL-USD`
- **한국주식**: `005930.KS`(삼성전자), `035720.KS`(카카오), `247540.KQ`(에코프로비엠) 또는 숫자만 `005930`

**승률 정의 (목표/손절 방식)**: 각 위치에서 매수한 뒤,
**목표수익(익절)에 먼저 닿으면 승리**, **손절에 먼저 닿으면 패배**로 집계해요.
최대 보유기간까지 둘 다 안 닿으면 만기 시점 수익률의 부호로 판정합니다.
- 승률 = 목표수익에 먼저 도달한 비율
- 평균 수익 = 청산 시 평균 수익률
- 최대 수익 = 보유 중 평균 최대 도달 수익률
- 평균 보유 = 청산까지 평균 걸린 거래일수

**구간 폭 vs 완충(중심 간격)**: 구간 폭은 한 행이 커버하는 범위,
완충은 행 사이 간격이에요. 완충 < 구간 폭이면 구간이 겹치는 **슬라이딩** 방식이라
위치 변화를 더 촘촘하게 볼 수 있어요 (예: 폭 10% / 완충 5%).
    """)

c_a, c_b, c_c = st.columns([3, 1, 1])
with c_a:
    ticker = st.text_input("종목 티커 / 기업명", value="TQQQ",
                           placeholder="예: AAPL, BTC-USD, 005930.KS, 삼성전자, 카카오")
with c_b:
    band_width = st.selectbox("구간 폭", [5, 10, 15, 20], index=1,
                              help="한 구간이 커버하는 범위 (예: 10%면 중심±5%)")
with c_c:
    step = st.selectbox("중심 간격 (완충)", [1, 2, 5, 10], index=2,
                        help="행을 얼마나 촘촘히 찍을지. 구간 폭보다 작으면 구간이 겹칩니다(슬라이딩).")

c1, c2, c3 = st.columns(3)
with c1:
    target_pct = st.number_input("🎯 목표수익 (익절 %)", min_value=1.0, max_value=100.0,
                                 value=10.0, step=1.0,
                                 help="이 수익률에 먼저 닿으면 승리(익절)")
with c2:
    stop_pct = st.number_input("🛑 손절 (완충 %)", min_value=1.0, max_value=100.0,
                               value=5.0, step=1.0,
                               help="이 손실률에 먼저 닿으면 패배(손절)")
with c3:
    max_hold_choice = st.selectbox("최대 보유기간", ["3개월(63일)", "6개월(126일)", "12개월(252일)", "24개월(504일)"],
                                   index=2,
                                   help="이 기간까지 목표·손절 둘 다 안 닿으면 만기 청산")

max_hold = {"3개월(63일)": 63, "6개월(126일)": 126, "12개월(252일)": 252, "24개월(504일)": 504}[max_hold_choice]

# --- 설정 설명 (항상 표시) ---
st.markdown("""
<div style="border:1px solid #444; border-radius:8px; padding:14px 18px; margin:8px 0 4px 0; font-size:13.5px; line-height:1.7;">
  <b>⚙️ 설정 항목 설명</b><br>
  <b>· 구간 폭</b> — 한 행(구간)이 커버하는 200일선 대비 범위예요.
  예를 들어 <b>10%</b>면 중심에서 ±5%(예: -5%~+5%)를 한 구간으로 봅니다.<br>
  <b>· 중심 간격 (완충)</b> — 표의 행을 얼마나 촘촘히 찍을지예요.
  <b>완충 &lt; 구간 폭</b>이면 인접 구간이 서로 겹치는 <b>슬라이딩</b> 방식이라
  위치 변화를 부드럽게 볼 수 있어요 (예: 폭 10% / 완충 5% → 중심이 5%씩 이동).
  완충을 구간 폭과 같게 하면 구간이 겹치지 않아요.<br>
  <b>· 🎯 목표수익 (익절)</b> — 매수 후 이 수익률에 <b>먼저</b> 닿으면 <b>승리(익절)</b>로 집계.<br>
  <b>· 🛑 손절</b> — 이 손실률에 <b>먼저</b> 닿으면 <b>패배(손절)</b>로 집계.<br>
  <b>· 최대 보유기간</b> — 목표·손절 둘 다 안 닿으면 이 기간에 만기 청산하고,
  그 시점 수익률의 부호(+/-)로 승패를 판정해요.
</div>
""", unsafe_allow_html=True)

run = st.button("🔍 전수조사", type="primary", use_container_width=True)

# --- 한글 기업명 → 종목코드 자동 변환 ---
resolved_ticker = ticker.strip() if ticker else ""
resolved_name = None
proceed = run

should_search_kr = run and ticker and (_has_korean(ticker) or not _looks_like_ticker(ticker))

if should_search_kr:
    with st.spinner("한국 종목명 검색 중..."):
        kind, payload, *rest = (*resolve_korean_name(ticker), None)

    if kind == "code":
        code, name = payload, rest[0]
        resolved_ticker = f"{code}.KS"  # yfinance/FDR 공용, FDR은 접미사 제거해서 사용
        resolved_name = name
        st.info(f"🇰🇷 '{ticker}' → **{name} ({code})** 로 변환했어요.")
    elif kind == "candidates":
        cands = payload
        st.warning(f"'{ticker}' 와 일치하는 종목이 여러 개예요. 선택해 주세요:")
        options = [f"{name} ({code}) · {market}" for code, name, market in cands]
        chosen = st.selectbox("종목 선택", options, key="kr_candidate")
        # 선택 확정 버튼
        if st.button("✅ 이 종목으로 조회", key="confirm_candidate"):
            idx = options.index(chosen)
            code, name, _ = cands[idx]
            resolved_ticker = f"{code}.KS"
            resolved_name = name
            proceed = True
        else:
            proceed = False  # 아직 선택 대기
    else:  # kind == "none"
        if _has_korean(ticker):
            # 한글로 검색했는데 못 찾음 -> 한국종목 의도가 명확하니 에러
            st.error(f"'{ticker}' 에 해당하는 한국 종목을 찾지 못했어요. "
                     f"정식 상장명이나 종목코드(예: 005930)로 시도해 보세요.")
            proceed = False
        # 영문인데 못 찾음 -> 해외 티커일 수 있으니 원본 그대로 yfinance 시도
        # (resolved_ticker 는 원본 유지, proceed 도 run 값 그대로)

if proceed and resolved_ticker:
    ticker = resolved_ticker  # 이후 로직은 변환된 티커 사용
    with st.spinner(f"{ticker} 데이터 로딩 중..."):
        raw = load_prices(ticker)

    if raw.empty:
        st.error("데이터를 찾을 수 없어요. 티커를 확인해 주세요.")
    elif len(raw) < 250:
        st.warning(f"데이터가 {len(raw)}일치뿐이라 200일선 분석이 어려워요 (최소 250일 필요).")
    else:
        df = prepare(raw)
        cur_gap = float(df["gap"].iloc[-1])
        cur_price = float(df["Close"].iloc[-1])
        cur_sma = float(df["SMA200"].iloc[-1])
        last_date = df.index[-1].strftime("%Y-%m-%d")
        total_days = len(df)

        # --- 현재 위치 요약 ---
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        display_name = resolved_name if resolved_name else ticker.upper()
        c1.metric("종목", display_name)
        c2.metric("현재가", f"{cur_price:,.2f}")
        c3.metric("200일선", f"{cur_sma:,.2f}")
        gap_display = f"{cur_gap:+.1f}%"
        c4.metric("200일선 대비", gap_display)

        st.markdown(f"데이터: **{total_days}** 거래일 | 🎯 목표 **+{target_pct:.0f}%** / "
                    f"🛑 손절 **-{stop_pct:.0f}%** | 최대보유: **{max_hold_choice}** | "
                    f"구간 폭: **{band_width}%** / 완충: **{step}%** | 기준일: {last_date}")

        # --- 구간별 전수조사 ---
        with st.spinner("전수조사 계산 중..."):
            result = zone_analysis(df, band_width, step, target_pct, stop_pct, max_hold)

        if result.empty:
            st.warning("분석할 데이터가 부족합니다.")
        else:
            # 현재 위치 = center가 현재 괴리율에 가장 가까운 구간
            # (슬라이딩이라 현재 gap을 포함하는 구간이 여럿일 수 있음)
            cur_zone_idx = int((result["center"] - cur_gap).abs().idxmin())

            st.markdown(f"### 📊 [핵심] 200일선 대비 위치별 승률")
            st.markdown(f"폭 {band_width}%, 완충 {step}%, 목표 +{target_pct:.0f}% / 손절 -{stop_pct:.0f}% "
                        f"(먼저 닿는 쪽), 최대보유 {max_hold_choice} 기준 전수조사 결과:")

            # HTML 테이블 생성
            html = '<table class="position-table">'
            html += """<tr>
                <th>중심 위치</th>
                <th>구간</th>
                <th>거래수</th>
                <th>승률</th>
                <th>평균 수익</th>
                <th>최대 수익</th>
                <th>평균 보유</th>
            </tr>"""

            for i, row in result.iterrows():
                is_current = (i == cur_zone_idx)
                row_class = ' class="current-row"' if is_current else ''

                # 승률 색상
                wr = row["win_rate"]
                if wr >= 60:
                    wr_cls = "win-high"
                elif wr >= 45:
                    wr_cls = "win-mid"
                else:
                    wr_cls = "win-low"

                center_label = f"{row['center']:+.0f}%"
                if abs(row["center"]) < 0.01:
                    center_label = "0% (200일선)"

                marker = " ◀ 현재" if is_current else ""

                html += f"""<tr{row_class}>
                    <td>{center_label}{marker}</td>
                    <td>{row['zone_label']}</td>
                    <td>{row['trades']}</td>
                    <td class="{wr_cls}">{wr:.0f}%</td>
                    <td>{row['avg_return']:+.1f}%</td>
                    <td>{row['max_return']:+.1f}%</td>
                    <td>{row['avg_holding_days']}일</td>
                </tr>"""

            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)

            # --- 현재 위치 결론 ---
            if cur_zone_idx is not None:
                cur_row = result.iloc[cur_zone_idx]
                wr = cur_row["win_rate"]
                rule = f"목표 +{target_pct:.0f}% / 손절 -{stop_pct:.0f}%, 표본 {cur_row['trades']}건"
                st.markdown("---")
                if wr >= 60:
                    st.success(
                        f"🟢 현재 위치({cur_gap:+.1f}%)에서 매수하면 **목표 도달 확률이 높았어요**. "
                        f"승률 **{wr:.0f}%**, 평균 청산수익 **{cur_row['avg_return']:+.1f}%**, "
                        f"평균 보유 **{cur_row['avg_holding_days']}일** ({rule})"
                    )
                elif wr >= 45:
                    st.warning(
                        f"🟡 현재 위치({cur_gap:+.1f}%)는 목표·손절이 반반이에요. "
                        f"승률 **{wr:.0f}%**, 평균 청산수익 **{cur_row['avg_return']:+.1f}%**, "
                        f"평균 보유 **{cur_row['avg_holding_days']}일** ({rule})"
                    )
                else:
                    st.error(
                        f"🔴 현재 위치({cur_gap:+.1f}%)에서는 **손절 확률이 더 높았어요**. "
                        f"승률 **{wr:.0f}%**, 평균 청산수익 **{cur_row['avg_return']:+.1f}%**, "
                        f"평균 보유 **{cur_row['avg_holding_days']}일** ({rule})"
                    )

            # --- 가격 차트 (항상 표시) ---
            st.markdown("### 📉 가격 vs 200일선")
            chart_df = df[["Close", "SMA200"]].rename(columns={"Close": "종가", "SMA200": "200일선"})
            st.line_chart(chart_df)

st.divider()
st.caption("⚠️ 과거 데이터 기반 통계이며 투자 권유가 아닙니다. 과거 성과가 미래를 보장하지 않습니다.")
