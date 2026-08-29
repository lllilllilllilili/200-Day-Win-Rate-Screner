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

# ------------------------------------------------------------
# 즐겨찾기 (URL 쿼리 파라미터에 저장 — 북마크/홈화면 추가로 유지)
# ------------------------------------------------------------
# URL 형식: ?fav=TQQQ|TQQQ,005930.KS|삼성전자
#   각 항목은 "티커|이름", 항목 구분은 콤마.
_FAV_PARAM = "fav"


def get_favorites() -> list:
    """URL 쿼리 파라미터에서 즐겨찾기 목록 [{'ticker','name'}, ...] 읽기."""
    try:
        raw = st.query_params.get(_FAV_PARAM, "")
    except Exception:
        raw = ""
    if not raw:
        return []
    favs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "|" in item:
            ticker, name = item.split("|", 1)
        else:
            ticker, name = item, item
        ticker = ticker.strip()
        name = name.strip() or ticker
        if ticker and not any(f["ticker"] == ticker for f in favs):
            favs.append({"ticker": ticker, "name": name})
    return favs


def _persist_favs(favs: list):
    """즐겨찾기를 URL 쿼리 파라미터에 기록."""
    try:
        if favs:
            encoded = ",".join(f"{f['ticker']}|{f['name']}" for f in favs)
            st.query_params[_FAV_PARAM] = encoded
        else:
            # 비면 파라미터 제거
            if _FAV_PARAM in st.query_params:
                del st.query_params[_FAV_PARAM]
    except Exception:
        pass


def add_favorite(ticker: str, name: str):
    favs = list(get_favorites())
    if not any(f.get("ticker") == ticker for f in favs):
        favs.append({"ticker": ticker, "name": name})
        _persist_favs(favs)
    return favs


def remove_favorite(ticker: str):
    favs = [f for f in get_favorites() if f.get("ticker") != ticker]
    _persist_favs(favs)
    return favs


def is_favorite(ticker: str) -> bool:
    return any(f.get("ticker") == ticker for f in get_favorites())

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
@st.cache_data(ttl=86400, show_spinner=False, max_entries=1)
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


@st.cache_data(ttl=3600, show_spinner=False, max_entries=40)
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
        df["Close"] = df["Close"].astype("float32")  # 메모리 절감
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
        df["Close"] = df["Close"].astype("float32")  # 메모리 절감
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

    # --- 1단계: 각 진입 시점(pos)의 결과를 딱 한 번만 계산 ---
    # (target/stop/max_hold 가 고정이면 결과는 gap 과 무관하므로, 구간별로 재계산할 필요 없음.
    #  슬라이딩 구간이 겹쳐 같은 pos 가 여러 구간에 들어가도 계산은 1회만 하고 인덱싱으로 집계.)
    exit_ret = np.full(n, np.nan)   # 청산 수익률(%)
    max_ret = np.full(n, np.nan)    # 보유 중 최대 도달 수익률(%)
    hold_day = np.zeros(n, dtype=np.int32)
    win = np.zeros(n, dtype=bool)

    for pos in range(n - 1):
        entry = close[pos]
        end = min(pos + max_hold, n - 1)
        path = close[pos + 1:end + 1]
        if path.size == 0:
            continue
        cum = (path / entry - 1.0) * 100.0
        max_ret[pos] = cum.max()

        hit_t = np.argmax(cum >= target_pct) if (cum >= target_pct).any() else -1
        hit_s = np.argmax(cum <= -stop_pct) if (cum <= -stop_pct).any() else -1

        if hit_t != -1 and (hit_s == -1 or hit_t <= hit_s):
            win[pos] = True
            exit_ret[pos] = target_pct
            hold_day[pos] = hit_t + 1
        elif hit_s != -1:
            exit_ret[pos] = -stop_pct
            hold_day[pos] = hit_s + 1
        else:
            final = cum[-1]
            win[pos] = final > 0
            exit_ret[pos] = final
            hold_day[pos] = path.size

    valid = ~np.isnan(exit_ret)  # 결과가 있는 진입 시점

    # --- 2단계: 슬라이딩 구간별로 인덱싱 집계 ---
    n_neg = int(np.floor((0 - zone_min) / step))
    n_pos = int(np.floor((zone_max - 0) / step))
    centers = [round(-k * step, 6) for k in range(n_neg, 0, -1)] + \
              [round(k * step, 6) for k in range(0, n_pos + 1)]

    def fmt(v):
        return f"{v:+.0f}" if abs(v - round(v)) < 1e-9 else f"{v:+.1f}"

    rows = []
    for center in centers:
        lo, hi = center - half, center + half
        sel = valid & (gap >= lo) & (gap < hi)
        trades = int(sel.sum())
        if trades == 0:
            continue
        rows.append({
            "center": center,
            "zone_label": f"{fmt(lo)}%~{fmt(hi)}%",
            "trades": trades,
            "win_rate": float(win[sel].mean() * 100),
            "avg_return": float(exit_ret[sel].mean()),
            "max_return": float(np.nanmean(max_ret[sel])),
            "avg_holding_days": int(round(hold_day[sel].mean())),
        })

    return pd.DataFrame(rows)


# ============================================================
# 추가 도구 1: 크립토 200일선 × MVRV 스크리너
# ============================================================
import urllib.request
import json as _json
from datetime import timedelta

_CRYPTO_ASSETS = [
    {'ticker': 'BTC-USD', 'name': 'BTC (비트코인)', 'short': 'BTC', 'buffer': 0.06, 'has_mvrv': True},
    {'ticker': 'ETH-USD', 'name': 'ETH (이더리움)', 'short': 'ETH', 'buffer': 0.06, 'has_mvrv': False},
]
_MVRV_ZONES = {
    'strong_buy': (0, 1.0), 'buy': (1.0, 1.5), 'neutral': (1.5, 2.0),
    'caution': (2.0, 2.5), 'overheated': (2.5, 3.0), 'extreme': (3.0, float('inf')),
}
_MVRV_LABEL = {
    'strong_buy': ('🟢🟢', '강력 매수 구간 (역사적 바닥)'),
    'buy': ('🟢', '매수 적극 (저평가)'),
    'neutral': ('🟡', '관망/소량 매수 (중립)'),
    'caution': ('🟠', '매수 자제 (과열 시작)'),
    'overheated': ('🔴', '부분 익절 고려 (과열)'),
    'extreme': ('🔴🔴', '적극 익절 (극도 과열)'),
}


@st.cache_data(ttl=3600, show_spinner=False, max_entries=1)
def _load_btc_mvrv():
    try:
        url = "https://bitcoin-data.com/v1/mvrv"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode())
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['d'])
        return df.set_index('date')[['mvrv']].sort_index()
    except Exception:
        return None


def _mvrv_zone(v):
    for zone, (lo, hi) in _MVRV_ZONES.items():
        if lo <= v < hi:
            return zone
    return 'extreme'


def render_crypto_screener():
    st.subheader("🪙 크립토 200일선 × MVRV 스크리너")
    st.caption("BTC·ETH의 200일선 상태와 BTC MVRV(온체인 밸류에이션)를 함께 봅니다.")

    if not st.button("🔍 크립토 스캔", type="primary", key="crypto_scan"):
        st.info("버튼을 눌러 최신 BTC·ETH 상태를 스캔하세요.")
        return

    with st.spinner("데이터 로딩 중..."):
        btc_mvrv = _load_btc_mvrv()
        rows = []
        for asset in _CRYPTO_ASSETS:
            raw = load_prices(asset['ticker'])
            if raw.empty or len(raw) < 200:
                continue
            d = raw.copy()
            d["SMA200"] = d["Close"].rolling(200).mean()
            d = d.dropna()
            price = float(d["Close"].iloc[-1])
            sma = float(d["SMA200"].iloc[-1])
            gap = (price / sma - 1) * 100
            above = price > sma
            sell_line = sma * (1 - asset['buffer'])
            rows.append({
                "자산": asset['short'], "현재가": price, "200일선": sma,
                "괴리율": gap, "위/아래": "위" if above else "아래",
                "매도선": sell_line, "이탈": price < sell_line,
            })

    if not rows:
        st.error("크립토 데이터를 불러오지 못했어요.")
        return

    # MVRV 요약
    if btc_mvrv is not None and len(btc_mvrv) > 0:
        mv = float(btc_mvrv.iloc[-1]["mvrv"])
        zone = _mvrv_zone(mv)
        emoji, label = _MVRV_LABEL[zone]
        mv_date = btc_mvrv.index[-1].strftime("%Y-%m-%d")
        st.markdown(f"**BTC MVRV**: {mv:.3f} {emoji} — {label}  \n"
                    f"<span style='color:gray'>기준일 {mv_date} · MVRV 1.5 이하면 크립토 전반 저평가</span>",
                    unsafe_allow_html=True)
    else:
        st.warning("BTC MVRV 데이터를 불러오지 못했어요 (외부 API). 가격/200일선 정보만 표시합니다.")

    # 대시보드 테이블
    table = pd.DataFrame([{
        "자산": r["자산"],
        "현재가": f"{r['현재가']:,.2f}",
        "200일선": f"{r['200일선']:,.2f}",
        "괴리율": f"{r['괴리율']:+.1f}%",
        "위/아래": r["위/아래"],
        "매도선(6%완충)": f"{r['매도선']:,.2f}",
        "상태": "🚨 이탈" if r["이탈"] else ("✅ 위" if r["위/아래"] == "위" else "⏸️ 아래"),
    } for r in rows])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("⚠️ 과거 데이터 기반 분석이며 투자 권유가 아닙니다.")


# ============================================================
# 추가 도구 2: 시장 붕괴 경고 스캐너 (간소화)
# ============================================================
def _safe_dl(ticker, period="1y"):
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty:
            return None
        # 최근일 종가가 아직 미확정(nan)인 경우가 있어 Close 기준으로 정리
        if "Close" in data.columns:
            data = data.dropna(subset=["Close"])
        return None if data.empty else data.sort_index()
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False, max_entries=1)
def _crash_indicators():
    """시장 데이터 기반 지표들을 계산. (외부 매크로 지표는 프록시/근사)"""
    out = []

    # VIX
    vix = _safe_dl("^VIX")
    if vix is not None and not vix.empty:
        v = float(vix["Close"].iloc[-1])
        avg = float(vix["Close"].mean())
        if v < 12: score, stt = 60, "경계"
        elif v < 15: score, stt = 40, "경계"
        elif v < 20: score, stt = 20, "정상"
        elif v < 25: score, stt = 30, "정상"
        elif v < 30: score, stt = 50, "경계"
        else: score, stt = 90, "점등"
        out.append(("VIX (공포지수)", f"{v:.1f} (1년평균 {avg:.1f})", stt, score,
                    "극단적 저(안일)/고(패닉) 모두 경고"))

    # 수익률 곡선 10Y-3M
    tnx, irx = _safe_dl("^TNX"), _safe_dl("^IRX")
    if tnx is not None and irx is not None and not tnx.empty and not irx.empty:
        spread = float(tnx["Close"].iloc[-1]) - float(irx["Close"].iloc[-1])
        if spread < -0.5: score, stt = 60, "점등"
        elif spread < 0: score, stt = 70, "점등"
        elif spread < 0.3: score, stt = 80, "경계"
        elif spread < 0.8: score, stt = 50, "경계"
        else: score, stt = 20, "정상"
        out.append(("수익률 곡선 (10Y-3M)", f"{spread:.2f}%", stt, score,
                    "역전 후 정상화 시 침체 임박 신호"))

    # 시장 폭: SPY vs RSP (6개월)
    spy6, rsp6 = _safe_dl("SPY", "6mo"), _safe_dl("RSP", "6mo")
    if spy6 is not None and rsp6 is not None and not spy6.empty and not rsp6.empty:
        sr = (float(spy6["Close"].iloc[-1]) / float(spy6["Close"].iloc[0]) - 1) * 100
        rr = (float(rsp6["Close"].iloc[-1]) / float(rsp6["Close"].iloc[0]) - 1) * 100
        div = sr - rr
        ad = abs(div) if div > 0 else 0
        if ad <= 2: score = 10
        elif ad >= 15: score = 100
        else: score = min(100, int((ad - 2) / 13 * 100))
        stt = "점등" if div >= 10 else ("경계" if div >= 5 else "정상")
        out.append(("시장 폭 (SPY vs RSP, 6M)", f"{div:+.1f}%p", stt, score,
                    "괴리 클수록 소수 대형주 의존"))

    # S&P500 vs 200일선
    spy = _safe_dl("SPY")
    if spy is not None and not spy.empty and len(spy) >= 200:
        cur = float(spy["Close"].iloc[-1])
        sma = float(spy["Close"].iloc[-200:].mean())
        pct = (cur - sma) / sma * 100
        if pct < -5: score, stt = 90, "점등"
        elif pct < 0: score, stt = 60, "경계"
        elif pct > 15: score, stt = 40, "경계"
        else: score, stt = 15, "정상"
        out.append(("S&P500 vs 200일선", f"{pct:+.1f}%", stt, score,
                    "200일선 이탈 = 하락 추세, 기관 매도 트리거"))

    return out


def render_crash_scanner():
    st.subheader("🌡️ 시장 붕괴 경고 스캐너")
    st.caption("시장 데이터 기반 위험 지표를 종합해 위험도를 산출합니다. (매크로 지표는 프록시/근사)")

    if not st.button("🔍 위험도 스캔", type="primary", key="crash_scan"):
        st.info("버튼을 눌러 현재 시장 위험 지표를 스캔하세요.")
        return

    with st.spinner("시장 데이터 수집 중... (약 10~30초)"):
        inds = _crash_indicators()

    if not inds:
        st.error("데이터를 불러오지 못했어요. 잠시 후 다시 시도해 주세요.")
        return

    scores = [i[3] for i in inds]
    overall = int(np.mean(scores)) if scores else 0
    lit = sum(1 for i in inds if i[2] == "점등")
    caution = sum(1 for i in inds if i[2] == "경계")
    normal = sum(1 for i in inds if i[2] == "정상")

    if overall >= 75:
        level, rec, box = "위험", "주식 비중 대폭 축소, 현금/국채/금 확대", st.error
    elif overall >= 55:
        level, rec, box = "경계", "신규 매수 자제, 레버리지 해제, 현금 20~30% 확보", st.warning
    elif overall >= 35:
        level, rec, box = "주의", "포트폴리오 점검, 분산 유지, 급등주 차익 실현 검토", st.warning
    else:
        level, rec, box = "안전", "정상 투자 유지, 장기 매수 전략 지속", st.success

    box(f"**종합 위험도 {overall}/100 [{level}]** · 점등 {lit} / 경계 {caution} / 정상 {normal}  \n권고: {rec}")

    mark = {"점등": "🔴", "경계": "🟡", "정상": "🟢"}
    table = pd.DataFrame([{
        "지표": name, "값": val, "상태": f"{mark.get(stt,'⚪')} {stt}",
        "점수": score, "설명": desc,
    } for name, val, stt, score, desc in inds])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("⚠️ 일부 매크로 지표는 시장 데이터 기반 근사치입니다. 참고용이며 투자 권유가 아닙니다.")


# ============================================================
# 추가 도구 3: 데일리 스캐너 (200일선 상태 모니터)
# ============================================================
_DS_INDICES = {'QQQ': '나스닥100', 'SPY': 'S&P500', '069500.KS': '코스피(ETF)'}
_DS_US_TIER1 = ['NVDA', 'META', 'GOOGL', 'AAPL', 'CRM', 'MA', 'CSCO']
_DS_ALT = ['ETH-USD', 'SOL-USD', 'DOGE-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD', 'LINK-USD', 'BNB-USD']
_DS_KR = {
    '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '005380.KS': '현대차',
    '000270.KS': '기아', '051910.KS': 'LG화학', '105560.KS': 'KB금융',
    '030200.KS': 'KT', '055550.KS': '신한지주',
}


@st.cache_data(ttl=1800, show_spinner=False, max_entries=150)
def _ds_status(ticker):
    """종가/200SMA/괴리/돌파신호 반환. (30분 캐시로 반복 다운로드 방지)"""
    data = _safe_dl(ticker, "2y")
    if data is None or len(data) < 201:
        return None
    data = data.copy()
    data["MA200"] = data["Close"].rolling(200).mean()
    data = data.dropna(subset=["MA200"])
    if len(data) < 2:
        return None
    close = float(data["Close"].iloc[-1])
    ma = float(data["MA200"].iloc[-1])
    pclose = float(data["Close"].iloc[-2])
    pma = float(data["MA200"].iloc[-2])
    gap = (close / ma - 1) * 100
    signal = "-"
    if pclose <= pma and close > ma:
        signal = "🟢 BUY (돌파)"
    elif pclose >= pma and close < ma:
        signal = "🔴 SELL (이탈)"
    return {"close": close, "ma": ma, "gap": gap, "above": close > ma, "signal": signal}


def _ds_table(items):
    rows = []
    for ticker, name in items:
        try:
            r = _ds_status(ticker)
        except Exception:
            r = None  # 종목 하나가 실패해도 전체 스캔은 계속
        if not r:
            continue
        rows.append({
            "종목": name, "종가": f"{r['close']:,.2f}", "200SMA": f"{r['ma']:,.2f}",
            "괴리율": f"{r['gap']:+.1f}%", "위/아래": "위" if r["above"] else "아래",
            "신호": r["signal"],
        })
    return pd.DataFrame(rows)


# --- 국장 대형주 "200일선 아래 매수" 전략 승률 (전수조사 결과 내장) ---
# 각 값: (티커, 이름, {-5, -10, -15, -20 구간별 승률%}, 타입)
# 타입 A=얕게(-5~10 스윗스팟), B=깊이(-15~20 최적), C=비추
_KR_WINZONE = [
    ("012330.KS", "현대모비스", {5: 86, 10: 93, 15: 100, 20: 100}, "B"),
    ("005930.KS", "삼성전자", {5: 90, 10: 90, 15: 92, 20: 86}, "A"),
    ("003670.KS", "포스코퓨처엠", {5: 84, 10: 82, 15: 93, 20: 91}, "B"),
    ("010130.KS", "고려아연", {5: 86, 10: 79, 15: 80, 20: 100}, "B"),
    ("028260.KS", "삼성물산", {5: 80, 10: 82, 15: 100, 20: 100}, "B"),
    ("030200.KS", "KT", {5: 84, 10: 94, 15: 90, 20: 80}, "A"),
    ("017670.KS", "SK텔레콤", {5: 89, 10: 82, 15: 89, 20: 80}, "A"),
    ("000270.KS", "기아", {5: 88, 10: 81, 15: 75, 20: 88}, "A"),
    ("005380.KS", "현대차", {5: 78, 10: 70, 15: 79, 20: 90}, "B"),
    ("006400.KS", "삼성SDI", {5: 83, 10: 82, 15: 87, 20: 83}, "A"),
    ("051910.KS", "LG화학", {5: 77, 10: 80, 15: 88, 20: 83}, "B"),
    ("112040.KQ", "위메이드", {5: 78, 10: 77, 15: 81, 20: 85}, "B"),
    ("086520.KQ", "에코프로", {5: 88, 10: 83, 15: 76, 20: 69}, "A"),
    ("247540.KQ", "에코프로비엠", {5: 92, 10: 86, 15: 80, 20: 67}, "A"),
    ("000810.KS", "삼성화재", {5: 92, 10: 88, 15: None, 20: None}, "A"),
    ("086790.KS", "하나금융", {5: 91, 10: 90, 15: None, 20: None}, "A"),
    ("033780.KS", "KT&G", {5: 91, 10: 91, 15: None, 20: None}, "A"),
    ("035420.KS", "네이버", {5: 83, 10: 69, 15: 62, 20: 50}, "A"),
    ("035720.KS", "카카오", {5: 83, 10: None, 15: None, 20: None}, "A"),
    ("015760.KS", "한국전력", {5: None, 10: None, 15: 79, 20: 100}, "B"),
    ("196170.KQ", "알테오젠", {5: None, 10: None, 15: None, 20: 86}, "B"),
    ("000660.KS", "SK하이닉스", {5: 5, 10: 3, 15: None, 20: None}, "C"),
    ("293490.KQ", "카카오게임즈", {5: 50, 10: None, 15: None, 20: None}, "C"),
]
_TYPE_LABEL = {
    "A": "🅰️ 얕게(-5 ~ -10%가 스윗스팟)",
    "B": "🅱️ 깊이(-15 ~ -20%가 최적)",
    "C": "🚫 비추(전략 안 맞음)",
}


def _nearest_zone(gap):
    """현재 괴리율이 어느 매수 구간에 해당하는지. gap은 음수(아래)일 때만."""
    if gap > -3:
        return None  # 아직 매수 구간 아님 (200일선 근처/위)
    for z in (20, 15, 10, 5):
        if gap <= -z:
            return z
    return None


def _best_zone(winrates: dict):
    """이 종목에서 승률이 가장 높은 매수 구간과 승률 반환. (zone, wr) 또는 None."""
    valid = [(z, wr) for z, wr in winrates.items() if wr is not None]
    if not valid:
        return None
    z, wr = max(valid, key=lambda x: x[1])
    return z, wr


def render_kr_winzone():
    st.markdown("#### 🇰🇷 국장 대형주 200일선 매수 전략 스캐너")
    st.caption("'200일선 아래 -N%에서 매수 → 200일선 복귀 시 매도' 전략. "
               "현재 위치와 그 구간의 역사적 승률을 함께 봅니다.")

    in_zone_rows = []   # 지금 매수 구간(-5% 이하)에 들어온 종목
    above_rows = []     # 200일선 위/근처 (대기) 종목
    for ticker, name, winrates, typ in _KR_WINZONE:
        r = _ds_status(ticker)
        if not r:
            continue
        gap = r["gap"]
        zone = _nearest_zone(gap)

        # 이 종목의 최고 승률 구간 (내려오면 어디가 제일 좋은지)
        best = _best_zone(winrates)
        best_str = f"-{best[0]}% ({best[1]}%)" if best else "-"
        pos = "위" if gap >= 0 else "아래"

        # 현재 매수 구간에 들어와 있고 그 구간 승률 데이터가 있으면 → 매수 구간 종목
        if zone is not None and winrates.get(zone) is not None:
            wr = winrates[zone]
            if wr >= 70:
                status = "🟢 매수 구간 (승률 높음)"
            elif wr >= 50:
                status = "🟡 매수 구간 (보통)"
            else:
                status = "🔴 매수 구간 (승률 낮음/비추)"
            in_zone_rows.append({
                "종목": name, "타입": typ,
                "현재 괴리율": f"{gap:+.1f}%",
                "현재 매수구간": f"-{zone}%",
                "구간 승률": f"{wr}%",
                "최고 승률 구간": best_str,
                "상태": status,
            })
        else:
            # 200일선 위이거나, 아래지만 그 구간 데이터가 없는 경우 → 대기 목록
            if gap >= 0:
                status = f"🔵 200일선 위 (+{gap:.1f}%) · 대기"
            elif gap > -5:
                status = f"⚪ 200일선 바로 아래 ({gap:.1f}%) · 매수 임박"
            else:
                status = f"⚫ 200일선 아래 ({gap:.1f}%) · 구간 데이터 없음"
            above_rows.append({
                "종목": name, "타입": typ,
                "현재 괴리율": f"{gap:+.1f}%",
                "현재 위치": f"200일선 {pos}",
                "최고 승률 구간": best_str,
                "상태": status,
            })

    if not in_zone_rows and not above_rows:
        st.warning("데이터를 불러오지 못했어요.")
        return

    # 1) 지금 매수 구간 종목
    if in_zone_rows:
        st.success(f"🎯 지금 매수 구간(-5% 이하)에 들어온 종목: **{len(in_zone_rows)}개**")
        st.dataframe(pd.DataFrame(in_zone_rows), use_container_width=True, hide_index=True)
    else:
        st.info("현재 매수 구간(-5% 이하)에 들어온 종목이 없어요. 아래는 200일선 위 대기 종목입니다.")

    # 2) 200일선 위/대기 종목 (숨기지 않고 항상 표시)
    if above_rows:
        st.markdown("**📈 200일선 위 / 대기 종목** — 지금은 매수 구간 아님. "
                    "'최고 승률 구간'은 이 종목이 그만큼 내려왔을 때 역사적으로 가장 승률이 높았던 지점이에요.")
        # 괴리율 낮은(200일선에 가까운/아래) 순으로 정렬해서 매수 임박 종목이 위로
        above_df = pd.DataFrame(above_rows)
        above_df["_sort"] = above_df["현재 괴리율"].str.rstrip("%").astype(float)
        above_df = above_df.sort_values("_sort").drop(columns="_sort")
        st.dataframe(above_df, use_container_width=True, hide_index=True)

    st.markdown(
        "**타입 A** 얕게(-5 ~ -10%가 스윗스팟) · **타입 B** 깊이(-15 ~ -20%가 최적) · "
        "**타입 C** 비추(SK하이닉스·카카오게임즈)  \n"
        "<span style='color:gray'>· '최고 승률 구간'은 그 종목이 200일선 아래 해당 지점까지 내려왔을 때 "
        "역사적으로 가장 높았던 매수 승률입니다 (Yahoo Finance 전체 기간 전수조사 내장값). "
        "200일선 위 매수의 승률이 아니라, '내려오면 여기가 기회'라는 참고 지표예요.</span>",
        unsafe_allow_html=True)


def render_favorites_section():
    """즐겨찾기한 종목들의 현재 200일선 상태 표시."""
    favs = get_favorites()
    st.markdown("#### ⭐ 내 즐겨찾기")
    if not favs:
        st.info("아직 즐겨찾기한 종목이 없어요. '위치별 승률 스크리너' 탭에서 종목 조회 후 "
                "**☆ 즐겨찾기 추가**를 누르면 여기에 표시됩니다.")
        return

    rows = []
    for f in favs:
        ticker, name = f.get("ticker"), f.get("name", f.get("ticker"))
        r = _ds_status(ticker)
        if not r:
            rows.append({"종목": name, "종가": "-", "200SMA": "-",
                         "괴리율": "-", "위/아래": "-", "신호": "데이터 없음"})
            continue
        rows.append({
            "종목": name, "종가": f"{r['close']:,.2f}", "200SMA": f"{r['ma']:,.2f}",
            "괴리율": f"{r['gap']:+.1f}%", "위/아래": "위" if r["above"] else "아래",
            "신호": r["signal"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 개별 삭제
    del_target = st.selectbox("삭제할 즐겨찾기", ["(선택)"] + [f["name"] for f in favs],
                              key="fav_del_select")
    if del_target != "(선택)":
        if st.button(f"🗑️ '{del_target}' 삭제", key="fav_del_btn"):
            tk = next((f["ticker"] for f in favs if f["name"] == del_target), None)
            if tk:
                remove_favorite(tk)
                st.rerun()

    st.caption(f"총 {len(favs)}개 즐겨찾기 · **URL에 저장**됩니다. "
               "이 페이지를 **북마크하거나 폰 홈화면에 추가**하면 다음에도 그대로 유지돼요.")


# ============================================================
# 추가 도구 5: 200일선 복귀 기간 (미국 시총 TOP50 전수 분석 내장값)
# ============================================================
# (티커, 이름, 사이클수, 평균일, 중앙값일, 최대일)
_RECOVERY_TOP50 = [
    ("AVGO", "AVGO", 78, 12, 4, 166), ("V", "V", 81, 14, 4, 164),
    ("LIN", "LIN", 142, 19, 4, 255), ("ABBV", "ABBV", 65, 20, 4, 406),
    ("PM", "PM", 88, 20, 6, 266), ("NEE", "NEE", 237, 20, 4, 420),
    ("CVX", "CVX", 347, 21, 5, 429), ("MA", "MA", 61, 21, 5, 238),
    ("AMGN", "AMGN", 220, 22, 5, 508), ("COST", "COST", 186, 22, 5, 337),
    ("BRK-B", "BRK-B", 122, 23, 4, 289), ("WMT", "WMT", 219, 24, 5, 294),
    ("TSLA", "TSLA", 83, 24, 7, 250), ("PEP", "PEP", 211, 25, 6, 433),
    ("ISRG", "ISRG", 101, 25, 5, 317), ("PG", "PG", 266, 25, 6, 387),
    ("MCD", "MCD", 251, 25, 5, 325), ("WFC", "WFC", 243, 25, 5, 369),
    ("MSFT", "MSFT", 153, 26, 5, 399), ("JNJ", "JNJ", 252, 26, 5, 375),
    ("ORCL", "ORCL", 163, 26, 6, 430), ("ABT", "ABT", 174, 26, 6, 316),
    ("ADBE", "ADBE", 158, 27, 4, 406), ("XOM", "XOM", 260, 27, 5, 497),
    ("KO", "KO", 240, 27, 6, 450), ("DHR", "DHR", 168, 27, 4, 498),
    ("GOOGL", "GOOGL", 67, 28, 4, 313), ("JPM", "JPM", 203, 28, 6, 475),
    ("CRM", "CRM", 71, 28, 7, 387), ("UNH", "UNH", 151, 29, 5, 602),
    ("HD", "HD", 159, 29, 5, 329), ("TMO", "TMO", 178, 29, 4, 515),
    ("AMZN", "AMZN", 111, 30, 5, 607), ("LLY", "LLY", 210, 31, 7, 592),
    ("META", "META", 34, 32, 6, 392), ("BAC", "BAC", 226, 32, 5, 395),
    ("MRK", "MRK", 227, 32, 6, 554), ("QCOM", "QCOM", 159, 33, 6, 537),
    ("TXN", "TXN", 237, 33, 6, 463), ("BRK-B", "BRK-B", 122, 23, 4, 289),
    ("MS", "MS", 127, 34, 6, 333), ("RTX", "RTX", 213, 34, 6, 453),
    ("AAPL", "AAPL", 132, 40, 6, 333), ("GS", "GS", 90, 40, 6, 353),
    ("NVDA", "NVDA", 61, 45, 5, 444), ("INTC", "INTC", 150, 45, 7, 416),
    ("NFLX", "NFLX", 52, 47, 6, 440), ("CSCO", "CSCO", 85, 48, 6, 418),
    ("AMD", "AMD", 151, 52, 6, 537), ("ACN", "ACN", 106, 24, 5, 302),
    ("INTU", "INTU", 131, 26, 5, 521),
]

# 전체 복귀 기간 분포 (누적 %)
_RECOVERY_DIST = [
    ("1일", "1,768회", "22.4%", "22.4%"),
    ("2~3일", "1,338회", "16.9%", "39.3%"),
    ("1주 이내", "1,580회", "20.0%", "59.3%"),
    ("1~2주", "933회", "11.8%", "71.1%"),
    ("2주~1개월", "743회", "9.4%", "80.5%"),
    ("1~2개월", "554회", "7.0%", "87.5%"),
    ("2~3개월", "297회", "3.8%", "91.3%"),
    ("3~6개월", "371회", "4.7%", "96.0%"),
    ("6개월~1년", "267회", "3.4%", "99.4%"),
    ("1~2년", "49회", "0.6%", "100%"),
]


def _recovery_action(median_days):
    """중앙값 복귀일 기준 행동 가이드."""
    if median_days <= 4:
        return "복귀 매우 빠름 → 내려오자마자 바로 매수 (모을 시간 짧음)"
    elif median_days <= 6:
        return "복귀 빠름 → 신속 분할 매수"
    else:
        return "복귀 다소 느림 → 여유 있게 분할 매수 가능"


# ============================================================
# 추가 도구 6: 미국 섹터 순환매 (1999~2026 계절성 내장값)
# ============================================================
_SECTOR_ETFS = {
    "XLK": "기술(IT)", "XLF": "금융", "XLE": "에너지", "XLV": "헬스케어",
    "XLY": "경기소비재", "XLP": "필수소비재", "XLI": "산업재", "XLB": "소재",
    "XLU": "유틸리티", "XLRE": "부동산", "XLC": "커뮤니케이션",
}

# 월별 (최강섹터, 수익률, 최약섹터, 수익률)
_MONTH_BEST_WORST = {
    1:  ("커뮤니케이션", "+4.2%", "소재", "-0.8%"),
    2:  ("에너지", "+1.6%", "유틸리티", "-1.0%"),
    3:  ("유틸리티", "+2.3%", "커뮤니케이션", "-0.8%"),
    4:  ("에너지", "+3.2%", "부동산", "+1.0%"),
    5:  ("커뮤니케이션", "+2.6%", "소재", "+0.2%"),
    6:  ("부동산", "+1.8%", "금융", "-0.8%"),
    7:  ("부동산", "+3.2%", "에너지", "+0.5%"),
    8:  ("커뮤니케이션", "+1.7%", "에너지", "-0.5%"),
    9:  ("유틸리티", "+0.1%", "부동산", "-3.0%"),
    10: ("기술(IT)", "+2.9%", "부동산", "-0.9%"),
    11: ("소재", "+3.6%", "유틸리티", "+0.8%"),
    12: ("소재", "+1.7%", "커뮤니케이션", "0%"),
}

# 월별 TOP3 + 꼴찌
_MONTH_TOP3 = {
    1:  ("커뮤니케이션", "에너지", "헬스케어", "소재"),
    2:  ("에너지", "소재", "산업재", "유틸리티"),
    3:  ("유틸리티", "에너지", "소재", "커뮤니케이션"),
    4:  ("에너지", "소재", "산업재", "부동산"),
    5:  ("커뮤니케이션", "기술(IT)", "유틸리티", "소재"),
    6:  ("부동산", "기술(IT)", "헬스케어", "금융"),
    7:  ("부동산", "커뮤니케이션", "금융", "에너지"),
    8:  ("커뮤니케이션", "기술(IT)", "부동산", "에너지"),
    9:  ("유틸리티", "필수소비재", "경기소비재", "부동산"),
    10: ("기술(IT)", "금융", "소재", "부동산"),
    11: ("소재", "커뮤니케이션", "산업재", "유틸리티"),
    12: ("소재", "헬스케어", "산업재", "커뮤니케이션"),
}

# 실전 순환매 액션 (매수 타이밍)
_SECTOR_ACTION = {
    1:  "커뮤니케이션(XLC) 보유 · 1월은 커뮤니케이션 +4.2%로 전 섹터/월 단일 최고",
    2:  "에너지(XLE)로 전환 · 봄 강세 시작. 유틸리티 축소",
    3:  "유틸리티·에너지 강세 · 봄 전력수요+드라이빙 시즌 기대",
    4:  "소재(XLB)·산업재(XLI) 매수 · 건설 착공/제조 가동. 에너지도 강세",
    5:  "커뮤니케이션·기술(IT) · 여름 앞두고 성장주",
    6:  "부동산(XLRE) 매수 · 6~7월 승률 91%로 전 섹터 최고",
    7:  "부동산 유지 · 주택 매매 성수기 지속",
    8:  "유틸리티(XLU)로 방어 전환 준비 · 9월 대비",
    9:  "유틸리티(XLU)만 · 9월은 유틸리티만 유일하게 플러스, 나머지 전부 약세",
    10: "기술(XLK) 매수 · 아이폰 출시+3분기 실적. 10월 말 소재·산업재도 담기",
    11: "전 섹터 강세월 · 소재(XLB) +3.6%(승률 81%) 최강. 아무거나 사도 3번 중 2번 수익",
    12: "소재·헬스케어·산업재 · 산타랠리. 커뮤니케이션은 이달 약세",
}

# 11월 전 섹터 강세 (수익률, 승률)
_NOV_STRENGTH = [
    ("소재", "+3.6%", "81%"), ("커뮤니케이션", "+3.5%", "75%"),
    ("산업재", "+3.5%", "78%"), ("부동산", "+3.1%", "64%"),
    ("경기소비재", "+2.9%", "78%"), ("기술(IT)", "+2.7%", "74%"),
    ("헬스케어", "+2.4%", "78%"), ("필수소비재", "+2.0%", "74%"),
    ("금융", "+2.0%", "67%"), ("에너지", "+1.8%", "59%"),
    ("유틸리티", "+0.8%", "59%"),
]

_MONTH_NAMES = {1:"1월",2:"2월",3:"3월",4:"4월",5:"5월",6:"6월",
                7:"7월",8:"8월",9:"9월",10:"10월",11:"11월",12:"12월"}


def render_sector_rotation():
    st.subheader("🗓️ 미국 섹터 순환매")
    st.caption("11개 미국 섹터 ETF의 월별 계절성 (1999~2026, 약 27년). 몇 월에 어떤 섹터가 강한지.")

    from datetime import datetime as _dt
    cur_month = _dt.now().month
    mname = _MONTH_NAMES[cur_month]

    # 이번 달 추천
    best, best_ret, worst, worst_ret = _MONTH_BEST_WORST[cur_month]
    top3 = _MONTH_TOP3[cur_month]
    st.success(f"### 📌 이번 달({mname}) 순환매\n"
               f"**최강 섹터: {best} ({best_ret})** · 최약: {worst} ({worst_ret})  \n"
               f"TOP3: {top3[0]} → {top3[1]} → {top3[2]} · 꼴찌: {top3[3]}")
    st.info(f"💡 **{mname} 액션**: {_SECTOR_ACTION[cur_month]}")

    # 11개 섹터 현재 200일선 상태
    if st.button("🔍 11개 섹터 현재 200일선 상태 스캔", type="primary", key="sector_scan"):
        with st.spinner("섹터 ETF 스캔 중..."):
            rows = []
            for tk, name in _SECTOR_ETFS.items():
                r = _ds_status(tk)
                if not r:
                    rows.append({"섹터": name, "ETF": tk, "현재가": "-",
                                 "200일선": "-", "괴리율": "-", "상태": "데이터 없음"})
                    continue
                stt = "🟢 200일선 위" if r["above"] else "🔴 200일선 아래"
                if r["signal"] != "-":
                    stt = r["signal"]
                rows.append({
                    "섹터": name, "ETF": tk,
                    "현재가": f"{r['close']:,.2f}", "200일선": f"{r['ma']:,.2f}",
                    "괴리율": f"{r['gap']:+.1f}%", "상태": stt,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 월별 순환매 캘린더
    st.markdown("#### 📅 순환매 캘린더 (매달 최강/최약)")
    cal = pd.DataFrame([{
        "월": _MONTH_NAMES[m],
        "최강 섹터": _MONTH_BEST_WORST[m][0], "수익률": _MONTH_BEST_WORST[m][1],
        "최약 섹터": _MONTH_BEST_WORST[m][2], "수익률 ": _MONTH_BEST_WORST[m][3],
        "TOP3": " · ".join(_MONTH_TOP3[m][:3]),
    } for m in range(1, 13)])
    # 이번 달 강조를 위해 마커
    cal["월"] = [f"👉 {_MONTH_NAMES[m]}" if m == cur_month else _MONTH_NAMES[m] for m in range(1, 13)]
    st.dataframe(cal, use_container_width=True, hide_index=True)

    # 계절별 패턴 + 특이점
    with st.expander("🌸 계절별 패턴 & 핵심 특이점", expanded=False):
        st.markdown("""
**계절별 주인공**
- **봄 (3~5월)**: 에너지 + 소재 + 산업재 (건설·제조·드라이빙 시즌 수요)
- **여름 (6~8월)**: 부동산(승률 91%) + IT + 커뮤니케이션
- **가을 (9~11월)**: IT + 소재 + 산업재 (아이폰·3분기 실적·내년 경기 기대)
- **겨울 (12~2월)**: 커뮤니케이션 + 에너지 (신년 광고비·한파)

**핵심 특이점**
- **9월**: 유틸리티(XLU)만 유일하게 플러스(+0.1%, 승률 63%). 나머지 10개 섹터 전부 약세. → 주식 하려면 유틸리티, 아니면 현금.
- **11월**: 11개 섹터 **전부 플러스**. 승률 70%+ 섹터가 8개. "아무거나 사도 3번 중 2번 수익". 10월 말 매수가 정답.
- **부동산**: 6~7월 승률 91%(전 섹터 최고), 9월 -3%(최악).
- **소재**: 11월 승률 81%(전 섹터/월 조합 TOP급).
- **커뮤니케이션**: 1월 +4.2%(단일 최고), 9월 -2.0%.
        """)

    with st.expander("🎊 11월은 왜 전 섹터가 오르나 (수익률/승률)", expanded=False):
        st.dataframe(pd.DataFrame(_NOV_STRENGTH, columns=["섹터", "11월 수익", "승률"]),
                     use_container_width=True, hide_index=True)
        st.markdown("세금 손실 매도(10월) 종료 후 재매수 · 산타랠리 · 선거 불확실성 해소 · "
                    "블랙프라이데이 소비 기대 · 기관 연말 성과 매수")

    st.caption("데이터: Yahoo Finance 섹터 ETF, 1999~2026 (내장값). "
               "XLC(2018)·XLRE(2015)는 표본이 짧음. 순환매는 확률이지 확정이 아니며, "
               "과거 패턴이 미래를 보장하지 않습니다.")


def render_recovery():
    st.subheader("🔄 200일선 복귀 기간")
    st.caption("미국 시총 TOP50이 200일선 아래로 내려간 뒤 얼마 만에 복귀했는지 전수 분석 (총 7,900 사이클).")

    st.info("📌 **핵심**: 절반은 **5일 이내** 복귀 · **60%가 1주 이내** · **80%가 1개월 이내** · 최악은 20개월(AMZN 닷컴버블). "
            "TOP50에서 2년 이상 200일선 아래 머문 종목은 **0건**.")

    # 전체 분포
    st.markdown("#### 📊 복귀 기간 분포 (전체 7,900 사이클)")
    st.dataframe(pd.DataFrame(_RECOVERY_DIST, columns=["기간", "횟수", "비율", "누적"]),
                 use_container_width=True, hide_index=True)

    # 종목별 복귀 통계 (정렬 옵션)
    st.markdown("#### 📈 종목별 복귀 통계")
    order = st.radio("정렬", ["복귀 빠른 순", "복귀 느린 순", "이름순"],
                     horizontal=True, key="recovery_order")
    # 중복 티커 제거
    seen = set()
    uniq = []
    for row in _RECOVERY_TOP50:
        if row[0] in seen:
            continue
        seen.add(row[0])
        uniq.append(row)

    if order == "복귀 빠른 순":
        uniq = sorted(uniq, key=lambda x: x[3])
    elif order == "복귀 느린 순":
        uniq = sorted(uniq, key=lambda x: -x[3])
    else:
        uniq = sorted(uniq, key=lambda x: x[0])

    table = pd.DataFrame([{
        "종목": name, "사이클": f"{cyc}회",
        "평균 복귀": f"{avg}일", "중앙값": f"{med}일",
        "최대(최악)": f"{mx}일 ({mx/30:.0f}개월)",
        "행동 가이드": _recovery_action(med),
    } for _, name, cyc, avg, med, mx in uniq])
    st.dataframe(table, use_container_width=True, hide_index=True)

    # 기간별 행동 가이드
    with st.expander("📖 200일선 아래 기간별 행동 가이드", expanded=False):
        st.markdown("""
| 아래 머문 기간 | 확률 | 행동 |
|---|---|---|
| 1~7일 | 60% | 소량 매수. 바로 올라올 수 있음 |
| 1~4주 | 21% | 기본 분할 매수. 가장 흔한 "모으기 구간" |
| 1~3개월 | 11% | 적극 매수. 할인 구간 확실 |
| 3~6개월 | 5% | 무겁게 매수. 진짜 기회 |
| 6개월~1년 | 3% | 최대 비중. 역사적 저가 |
| 1년 이상 | 0.6% | 종목 자체 문제 없는지 확인 필요 |

**핵심 3줄**
1. 60%는 1주 안에, 80%는 1개월 안에 복귀 — 대부분 "잠깐 찍고 반등"
2. 진짜 모을 기회(2주+)는 30% — 이때 분할 매수해야 의미 있는 물량 확보
3. 최악은 1~2년이지만 TOP50은 반드시 복귀 — 2년 이상 머문 종목 0건
        """)

    st.caption("데이터: Yahoo Finance 전체 기간 · 미국 시총 TOP50 · 총 7,900 이탈→복귀 사이클 (내장값). "
               "과거 성과가 미래를 보장하지 않습니다.")


def render_daily_screener():
    st.subheader("📋 데일리 스캐너")
    st.caption("지수·미국 대형주·국장 대표주·알트코인의 200일선 상태와 돌파/이탈 신호를 한 번에.")

    # 즐겨찾기는 버튼 없이 항상 상단에 표시 (있을 때만 스캔)
    if get_favorites():
        with st.spinner("즐겨찾기 상태 스캔 중..."):
            render_favorites_section()
        st.markdown("---")
    else:
        render_favorites_section()
        st.markdown("---")

    if not st.button("🔍 데일리 스캔", type="primary", key="daily_scan"):
        st.info("버튼을 눌러 오늘의 200일선 상태를 스캔하세요. (종목이 많아 20~40초 걸릴 수 있어요)")
        return

    with st.spinner("지수 스캔 중..."):
        st.markdown("#### 📊 지수")
        st.dataframe(_ds_table(list(_DS_INDICES.items())), use_container_width=True, hide_index=True)

    with st.spinner("미국 대형주 스캔 중..."):
        st.markdown("#### 🇺🇸 미국 주요주")
        st.dataframe(_ds_table([(t, t) for t in _DS_US_TIER1]), use_container_width=True, hide_index=True)

    with st.spinner("국장 대표주 스캔 중..."):
        st.markdown("#### 🇰🇷 국장 대표주")
        st.dataframe(_ds_table(list(_DS_KR.items())), use_container_width=True, hide_index=True)

    with st.spinner("국장 200일선 매수 전략 스캔 중..."):
        st.markdown("---")
        render_kr_winzone()
        st.markdown("---")

    with st.spinner("알트코인 스캔 중..."):
        st.markdown("#### 🪙 알트코인")
        st.dataframe(_ds_table([(t, t.replace("-USD", "")) for t in _DS_ALT]),
                     use_container_width=True, hide_index=True)

    st.caption("⚠️ 과거 데이터 기반이며 투자 권유가 아닙니다. BUY/SELL은 200일선 돌파/이탈 신호일 뿐입니다.")


# ============================================================
# 추가 도구 4: 아기티큐 TQQQ 200일선 전략 대시보드
# ============================================================
# 부분 익절 단계: (수익률 임계 %, 매도 비율 설명)
_BABYTQQQ_PROFIT_STEPS = [
    (10, "보유 수량의 10% 익절 → SPYM 전환"),
    (25, "보유 수량의 10% 익절 → SPYM 전환"),
    (50, "보유 수량의 10% 익절 → SPYM 전환"),
    (100, "남은 수량의 50% 익절 (대익절) → SPYM 전환"),
    (200, "남은 수량의 50% 익절 (대익절) → SPYM 전환"),
    (300, "남은 수량의 50% 익절 (대익절) · 이후 계속"),
]


# 로테이션 우선순위: TQQQ > BTC > SOXL (앞이 높은 우선순위)
_ROTATION_ASSETS = [
    {"ticker": "TQQQ", "name": "TQQQ", "buffer": 0.0, "prio": 1},
    {"ticker": "BTC-USD", "name": "BTC (3%완충)", "buffer": 0.03, "prio": 2},
    {"ticker": "SOXL", "name": "SOXL", "buffer": 0.0, "prio": 3},
]

# 백테스트 요약 (2015.07~2026.08, 초기 1000만+월 250만 적립) — 참고용 내장값
_ROTATION_BACKTEST = [
    {"전략": "갈아타기 (TQQQ>BTC>SOXL)", "배수": "15.2배", "CAGR": "+75.7%", "MDD": "-28.6%", "MAR": "2.65", "회복": "13개월"},
    {"전략": "유지 (신호 무시)", "배수": "13.6배", "CAGR": "+73.9%", "MDD": "-33.0%", "MAR": "2.24", "회복": "17개월"},
    {"전략": "TQQQ 단독", "배수": "7.6배", "CAGR": "+65.0%", "MDD": "-29.2%", "MAR": "2.23", "회복": "15개월"},
    {"전략": "BTC 단독 (3%완충)", "배수": "7.5배", "CAGR": "+64.8%", "MDD": "-27.1%", "MAR": "2.39", "회복": "16개월"},
    {"전략": "SOXL 단독", "배수": "11.4배", "CAGR": "+71.3%", "MDD": "-56.2%", "MAR": "1.27", "회복": "26개월"},
]


def _rotation_status(ticker, buffer):
    """로테이션용: 200일선 상태 + 완충 적용 매도선 이탈 여부."""
    raw = load_prices(ticker)
    if raw.empty or len(raw) < 200:
        return None
    d = raw.copy()
    d["SMA200"] = d["Close"].rolling(200).mean()
    d = d.dropna()
    if len(d) < 1:
        return None
    price = float(d["Close"].iloc[-1])
    sma = float(d["SMA200"].iloc[-1])
    gap = (price / sma - 1) * 100
    sell_line = sma * (1 - buffer)  # 완충 적용 매도선
    return {
        "price": price, "sma": sma, "gap": gap,
        "above": price > sma,
        "holdable": price >= sell_line,  # 완충 감안 보유 유지 가능
        "sell_line": sell_line,
        "date": d.index[-1].strftime("%Y-%m-%d"),
    }


def render_rotation():
    st.markdown("#### 🔄 로테이션 전략 (TQQQ > BTC > SOXL)")
    st.caption("우선순위 높은 종목이 200일선 위면 그쪽으로 갈아탑니다. "
               "TQQQ가 최우선, 없으면 BTC, 그것도 없으면 SOXL, 다 아래면 현금(SGOV).")

    if not st.button("🔍 로테이션 현재 상태 확인", type="primary", key="rotation_scan"):
        st.info("버튼을 눌러 3종목의 현재 200일선 상태와 '지금 어디 있어야 하는지'를 확인하세요.")
    else:
        with st.spinner("TQQQ · BTC · SOXL 상태 확인 중..."):
            statuses = {}
            for a in _ROTATION_ASSETS:
                statuses[a["ticker"]] = _rotation_status(a["ticker"], a["buffer"])

        if all(v is None for v in statuses.values()):
            st.error("데이터를 불러오지 못했어요.")
        else:
            # 우선순위대로 '보유 가능(200일선 위/완충 내)'인 첫 종목 선택
            target = None
            for a in _ROTATION_ASSETS:
                s = statuses.get(a["ticker"])
                if s and s["holdable"]:
                    target = a
                    break

            if target is None:
                st.warning("⏸️ **전부 200일선 아래 → 현금(SGOV) 대피 구간**  \n"
                           "→ 세 종목 모두 매도선 아래예요. SGOV에서 대기하세요.")
            else:
                s = statuses[target["ticker"]]
                st.success(f"🎯 **지금 보유해야 할 종목: {target['name']}**  \n"
                           f"→ 200일선 대비 {s['gap']:+.1f}%. 우선순위상 이 종목이 최상위 '보유 가능' 종목이에요.  \n"
                           f"→ 더 높은 우선순위 종목(위 순서)이 200일선을 돌파하면 그쪽으로 갈아타세요.")

            # 3종목 현황 테이블
            rows = []
            for a in _ROTATION_ASSETS:
                s = statuses.get(a["ticker"])
                if not s:
                    rows.append({"우선순위": a["prio"], "종목": a["name"],
                                 "현재가": "-", "200일선": "-", "괴리율": "-", "상태": "데이터 없음"})
                    continue
                if target and a["ticker"] == target["ticker"]:
                    stt = "🟢 보유 (현재 타겟)"
                elif s["holdable"]:
                    stt = "🟡 보유 가능 (하위 우선순위)"
                else:
                    stt = "🔴 200일선 아래 (제외)"
                rows.append({
                    "우선순위": a["prio"], "종목": a["name"],
                    "현재가": f"{s['price']:,.2f}", "200일선": f"{s['sma']:,.2f}",
                    "괴리율": f"{s['gap']:+.1f}%", "상태": stt,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"기준일 {statuses.get('TQQQ', {}).get('date', '-') if statuses.get('TQQQ') else '-'} "
                       "· BTC는 3% 완충 적용 (200일선 -3%까지 보유 유지)")

    # 백테스트 요약 (참고용 내장값)
    with st.expander("📊 백테스트 요약 (2015.07~2026.08, 참고용)", expanded=False):
        st.dataframe(pd.DataFrame(_ROTATION_BACKTEST), use_container_width=True, hide_index=True)
        st.markdown("""
- **갈아타기**가 최강 (배수 15.2배, MAR 2.65). 하위 종목 보유 중 상위가 돌파하면 즉시 전환.
- **TQQQ**가 리스크 대비 가장 효율적 (MAR·Sharpe 우수). **BTC(3%완충)**는 11년 전승. **SOXL**은 절대수익 1위지만 MDD -56%.
- **% 손절 금지**: 레버리지 ETF에 -3% 손절 넣으면 수익이 1/4로 축소. 200일선 이탈만이 매도 신호.
- **종가 기준만**: 장중 이탈로 판단하면 수익 1/16로 축소.
- 200일선 아래로 내려가도 **55%가 1주 이내, 80%가 1개월 이내** 복귀.

<span style='color:gray'>※ Yahoo Finance 기반, 수수료/세금/환율 미반영. 과거 성과가 미래를 보장하지 않습니다.</span>
        """, unsafe_allow_html=True)


def render_babytqqq():
    st.subheader("🍼 아기티큐 TQQQ 200일선 전략")
    st.caption("200일선 위=주식(TQQQ), 아래=채권(SGOV). 현재 상태와 다음 행동을 알려줍니다.")

    mode = st.radio("전략 선택", ["단일 TQQQ", "로테이션 (TQQQ>BTC>SOXL)"],
                    horizontal=True, key="baby_mode")
    st.markdown("---")
    if mode == "로테이션 (TQQQ>BTC>SOXL)":
        render_rotation()
        return
    st.markdown(
        "<span style='color:gray'>※ 커뮤니티에 공개된 'TQQQ 200일선 매매법'을 참고한 요약 도구입니다. "
        "투자 권유가 아니며 과거 성과가 미래를 보장하지 않습니다.</span>",
        unsafe_allow_html=True)

    if not st.button("🔍 TQQQ 현재 상태 확인", type="primary", key="baby_scan"):
        st.info("버튼을 눌러 TQQQ의 현재 200일선 상태와 전략 신호를 확인하세요.")
        _babytqqq_rules()
        return

    with st.spinner("TQQQ 데이터 로딩 중..."):
        raw = load_prices("TQQQ")

    if raw.empty or len(raw) < 200:
        st.error("TQQQ 데이터를 불러오지 못했어요.")
        _babytqqq_rules()
        return

    d = raw.copy()
    d["SMA200"] = d["Close"].rolling(200).mean()
    d = d.dropna()
    price = float(d["Close"].iloc[-1])
    sma = float(d["SMA200"].iloc[-1])
    gap = (price / sma - 1) * 100
    above = price > sma
    pclose = float(d["Close"].iloc[-2])
    pma = float(d["SMA200"].iloc[-2])
    last_date = d.index[-1].strftime("%Y-%m-%d")

    # 신호 판정
    just_crossed_up = pclose <= pma and price > sma
    just_crossed_down = pclose >= pma and price < sma

    c1, c2, c3 = st.columns(3)
    c1.metric("TQQQ 현재가", f"${price:,.2f}")
    c2.metric("200일선", f"${sma:,.2f}")
    c3.metric("200일선 대비", f"{gap:+.1f}%")

    if just_crossed_up:
        st.success("🟢 **매수 신호! (오늘 200일선 돌파)**  \n"
                   "→ 오늘부터 3일에 걸쳐 1/3씩 분할 매수 시작. "
                   "매수 도중 다시 200일선 아래로 내려가면 산 만큼만 즉시 매도하고 SGOV로 복귀.")
    elif just_crossed_down:
        st.error("🔴 **매도 신호! (오늘 200일선 이탈)**  \n"
                 "→ 애프터장(한국시간 새벽 5~8시)에 TQQQ 전량 매도 후 SGOV로 대피. "
                 "SPYM도 정리 대상.")
    elif above:
        st.info(f"✅ **보유 구간** — TQQQ가 200일선 위 (+{gap:.1f}%)에 있어요.  \n"
                "→ 보유 유지. 별도 손절 라인 없음. 200일선 이탈 시에만 매도. "
                "새 돈은 SPYM·SGOV 반반, 부분 익절은 아래 계산기 참고.")
    else:
        st.warning(f"⏸️ **대피 구간** — TQQQ가 200일선 아래 ({gap:.1f}%)에 있어요.  \n"
                   "→ SGOV(채권)에서 대기. 200일선 위로 재돌파할 때까지 TQQQ 매수 안 함. "
                   "새 돈은 SGOV만.")

    st.caption(f"기준일 {last_date} · TQQQ 종가/200일선 기준")

    # --- 부분 익절 계산기 ---
    st.markdown("---")
    st.markdown("#### 💰 부분 익절 목표 계산기")
    st.caption("내 평균 단가를 넣으면 각 익절 단계의 목표가를 알려줘요.")
    avg_price = st.number_input("내 TQQQ 평균 단가 ($)", min_value=0.0, value=float(round(price, 2)),
                                step=1.0, key="baby_avg")
    if avg_price > 0:
        cur_ret = (price / avg_price - 1) * 100
        st.markdown(f"현재 수익률: **{cur_ret:+.1f}%** (현재가 ${price:,.2f} / 평단 ${avg_price:,.2f})")
        rows = []
        next_target = None
        for thr, desc in _BABYTQQQ_PROFIT_STEPS:
            target_price = avg_price * (1 + thr / 100)
            reached = cur_ret >= thr
            if not reached and next_target is None:
                next_target = (thr, target_price)
            rows.append({
                "익절 단계": f"+{thr}%",
                "목표가": f"${target_price:,.2f}",
                "도달 여부": "✅ 도달" if reached else "⏳ 대기",
                "행동": desc,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if next_target:
            thr, tp = next_target
            need = (tp / price - 1) * 100
            st.info(f"🎯 다음 익절 목표: **+{thr}%** = **${tp:,.2f}** "
                    f"(현재가에서 {need:+.1f}% 더 오르면 도달)")
        else:
            st.success("🎉 모든 익절 단계를 이미 넘었어요. +300% 이후는 계속 대익절 규칙 적용.")

    _babytqqq_rules()


def _babytqqq_rules():
    with st.expander("📖 전략 규칙 요약", expanded=False):
        st.markdown("""
**핵심 원칙**: 200일선 **위 = 주식(TQQQ)**, **아래 = 채권(SGOV)**

**1. 매수 (200일선 돌파 시)**
- 3일에 걸쳐 1/3씩 분할 매수 (휩쏘 대비)
- 매수 도중 200일선 재이탈 시, 산 만큼만 즉시 매도 후 SGOV 복귀

**2. 매도 (200일선 이탈 시)**
- 이탈 당일 애프터장에 TQQQ 전량 매도 → SGOV 전환
- 별도 손절 라인 없음. 200일선 이탈이 유일한 매도 신호

**3. 새 돈이 생기면**
- 200일선 위: SPYM · SGOV 반반 (소액은 SPYM만)
- 200일선 아래: SGOV만

**4. 부분 익절 (수익률 기준, 최초 돌파 시)**
- 소익절: +10% / +25% / +50% → 각각 보유 수량의 10% 매도
- 대익절: +100% / +200% / +300% → 각각 남은 수량의 50% 매도
- 익절한 돈은 SPYM으로 전환 (한 방향: TQQQ → SPYM, 되돌리지 않음)
- SPYM은 200일선 이탈 때까지 보유

**5. 인출 순서**
- SGOV 먼저 → SPYM → TQQQ (TQQQ는 마지막)
        """)


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("📈 200일선 투자 도구 모음")

# --- 사이드바: 리소스 관리 ---
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    if st.button("🧹 캐시 비우기", help="앱이 느려지거나 리소스 경고가 뜨면 눌러 데이터 캐시를 초기화하세요."):
        st.cache_data.clear()
        st.success("캐시를 비웠어요. 다음 조회 시 최신 데이터를 다시 받아옵니다.")
    st.caption("데이터는 자동으로 캐시(30분~1시간)되고, 오래된 항목은 자동 정리됩니다. "
               "느려지면 위 버튼으로 캐시를 초기화하세요.")

# 대분류 (1depth) → 각 안에 하위 탭 (2depth)
group1, group2, group3, group4 = st.tabs([
    "🔍 종목 분석",
    "📊 시장 스캔",
    "📈 전략",
    "📚 통계·참고",
])

with group1:
    sub = st.tabs(["🎯 위치별 승률 스크리너", "🪙 크립토 200일선+MVRV"])
    tab1 = sub[0]
    with sub[1]:
        render_crypto_screener()

with group2:
    sub = st.tabs(["📋 데일리 스캐너", "🌡️ 시장 붕괴 경고"])
    with sub[0]:
        render_daily_screener()
    with sub[1]:
        render_crash_scanner()

with group3:
    sub = st.tabs(["🍼 아기티큐 TQQQ 전략"])
    with sub[0]:
        render_babytqqq()

with group4:
    sub = st.tabs(["🔄 200일선 복귀 기간", "🗓️ 섹터 순환매"])
    with sub[0]:
        render_recovery()
    with sub[1]:
        render_sector_rotation()

# ===== 탭 1: 기존 위치별 승률 스크리너 =====
tab1.markdown("**종목 검색 → 200일선 대비 모든 위치 구간의 역사적 승률을 한눈에**")
with tab1.expander("💡 사용법 & 티커 예시", expanded=False):
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

# 탭1의 모든 위젯을 tab1 컨텍스트에 렌더링
with tab1:
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
                                       index=0,
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
            raw_start = raw.index[0].strftime("%Y-%m-%d")   # 원본 데이터 시작일
            raw_end = raw.index[-1].strftime("%Y-%m-%d")    # 원본 데이터 마지막일
            analysis_start = df.index[0].strftime("%Y-%m-%d")  # 200일선 계산 후 분석 시작일
            raw_years = (raw.index[-1] - raw.index[0]).days / 365.25

            # --- 현재 위치 요약 ---
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            display_name = resolved_name if resolved_name else ticker.upper()
            c1.metric("종목", display_name)
            c2.metric("현재가", f"{cur_price:,.2f}")
            c3.metric("200일선", f"{cur_sma:,.2f}")
            gap_display = f"{cur_gap:+.1f}%"
            c4.metric("200일선 대비", gap_display)

            st.markdown(
                f"📅 **데이터 기간**: {raw_start} ~ {raw_end} "
                f"(약 {raw_years:.1f}년, {len(raw):,} 거래일)  \n"
                f"🔎 **분석 구간**: {analysis_start} ~ {last_date} "
                f"({total_days:,} 거래일) "
                f"<span style='color:gray'>· 앞 200일은 200일선 계산에 사용되어 분석에서 제외</span>",
                unsafe_allow_html=True)

            # --- 즐겨찾기 버튼 ---
            fav_ticker = ticker  # 변환된 최종 티커
            if is_favorite(fav_ticker):
                if st.button("⭐ 즐겨찾기 해제", key="unfav"):
                    remove_favorite(fav_ticker)
                    st.rerun()
                st.caption("⭐ 즐겨찾기됨 · 데일리 스캐너 탭에서 볼 수 있어요. "
                           "이 페이지를 북마크/홈화면에 추가하면 즐겨찾기가 유지돼요.")
            else:
                if st.button("☆ 즐겨찾기 추가", key="addfav"):
                    add_favorite(fav_ticker, display_name)
                    st.rerun()
                st.caption("☆ 추가하면 데일리 스캐너 탭에서 모아볼 수 있어요.")
            st.markdown(f"🎯 목표 **+{target_pct:.0f}%** / 🛑 손절 **-{stop_pct:.0f}%** | "
                        f"최대보유: **{max_hold_choice}** | "
                        f"구간 폭: **{band_width}%** / 완충: **{step}%**")

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
