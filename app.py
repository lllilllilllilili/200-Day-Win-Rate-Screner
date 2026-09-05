"""
200일선 위치별 승률 스크리너
================================
종목 검색 → 200일선 대비 위치를 구간별로 쪼개서
거래수 / 전략 성공률 / 평균수익 / 평균 최대도달 / 평균보유일수 를 전수조사 테이블로 보여준다.
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

# 위치별 승률 정적 테이블 스타일 (내부 스크롤 컨테이너 없음 = 모바일 스크롤 멈춤 방지)
st.markdown("""
<style>
.posbox { margin-top: 12px; }
.postbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    table-layout: fixed;
}
.postbl th, .postbl td {
    padding: 6px 4px;
    border-bottom: 1px solid #262a30;
    text-align: center;
    color: #d0d4d9;
    word-break: keep-all;
}
.postbl th {
    background: #1a1d23;
    color: #cbd2d9;
    font-weight: 600;
    font-size: 11px;
    line-height: 1.3;
}
.postbl th .sub { font-size: 10px; color: #7a828c; font-weight: 400; }
.postbl td.lft { text-align: left; }
.postbl td.dim { color: #9aa4af; }
.postbl td.price { color: #9aa4af; font-variant-numeric: tabular-nums; }
.postbl td .n { font-size: 10px; color: #6b7280; }
.postbl tr.cur td {
    background: rgba(59,130,246,0.20);
    font-weight: 700;
    box-shadow: inset 3px 0 0 #3b82f6;
}
.postbl td.wr-hi { color: #34d399; font-weight: 600; }
.postbl td.wr-mid { color: #fbbf24; }
.postbl td.wr-lo { color: #f87171; }
.postbl td.wr-na { color: #555; }
.postbl td.ret-up { color: #34d399; }
.postbl td.ret-dn { color: #f87171; }
.postbl td.ret-0 { color: #9aa4af; }
.postbl td.ret-max { color: #60a5fa; }
</style>
""", unsafe_allow_html=True)

# 승률 포착기 사전계산 데이터 (JSON 로딩 — .py 파싱보다 가볍고 안정적)
try:
    import json as _wz_json, os as _wz_os
    _wz_path = _wz_os.path.join(_wz_os.path.dirname(__file__), "winzone_data.json")
    with open(_wz_path, encoding="utf-8") as _wz_f:
        _wz = _wz_json.load(_wz_f)
    WINZONE_DATA = _wz["data"]
    WINZONE_META = _wz["meta"]
except Exception:
    WINZONE_DATA, WINZONE_META = {}, {"target": 10, "stop": 5, "max_hold": 63, "band": 10, "step": 5}

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


def _kr_ticker(code: str, market: str = None) -> str:
    """국내 종목코드에 시장별 yfinance 접미사를 붙인다.

    코스닥 종목을 '.KS'로 조회하면 yfinance가 동일 코드의 다른 상품(펀드 등)에
    매칭돼 재무 지표가 비어버리므로, 시장을 확인해 '.KQ'를 붙여야 한다.
    """
    code = str(code).strip().upper()
    code = code.replace(".KS", "").replace(".KQ", "")
    mk = market
    if mk is None:
        listing = load_krx_listing()
        if not listing.empty:
            hit = listing[listing["Code"].astype(str) == code]
            if len(hit):
                mk = str(hit.iloc[0]["Market"])
    return f"{code}.KQ" if (mk and "KOSDAQ" in mk.upper()) else f"{code}.KS"


def _normalize_kr_code(text: str) -> str:
    """'038500' 또는 '038500.KS'처럼 직접 입력한 국내 코드의 접미사를 시장에 맞게 교정."""
    import re
    s = text.strip().upper()
    m = re.fullmatch(r"(\d{6})(\.K[SQ])?", s)
    return _kr_ticker(m.group(1)) if m else text


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
    out["SMA50"] = out["Close"].rolling(50).mean()  # 추세 판정용
    out = out.dropna(subset=["SMA200"])
    out["gap"] = (out["Close"] - out["SMA200"]) / out["SMA200"] * 100
    return out


def zone_analysis(df: pd.DataFrame, band_width: float, step: float,
                  target_pct: float, stop_pct: float, max_hold: int,
                  zone_min: float = -50, zone_max: float = 100) -> pd.DataFrame:
    """
    200일선 괴리율을 슬라이딩 구간으로 나누고,
    각 구간에서 매수 후 "목표수익 도달 vs 손절 중 먼저 닿는 것"으로 성과를 전수조사.

    구간 생성 (슬라이딩 윈도우):
      - 중심(center)을 0%부터 step% 간격으로 양/음 방향으로 찍는다.
      - 각 중심마다 [center - band_width/2, center + band_width/2) 를 구간으로.
      - step < band_width 이면 인접 구간이 겹친다 (한 날이 여러 구간에 중복 집계).
      - step == band_width 이면 겹치지 않는다.

    판정:
      - 목표수익(target_pct%)에 먼저 닿으면 성공
      - 손절(stop_pct%)에 먼저 닿으면 실패
      - max_hold 거래일 안에 둘 다 안 닿으면, 만기 수익률이 양수일 때 성공
      - max_hold 미래 데이터가 온전히 남은 진입만 집계

    Returns DataFrame with primary/trend/alternative success rates, their sample counts,
    average exit return, exit-limited average MFE, and average holding days.
    """
    close = df["Close"].values
    gap = df["gap"].values
    sma = df["SMA200"].values
    sma50 = df["SMA50"].values if "SMA50" in df.columns else sma
    uptrend = sma50 > sma  # 추세: 50일선 > 200일선 = 상승추세
    n = len(close)
    half = band_width / 2

    # --- 1단계: 각 진입 시점(pos)의 결과를 딱 한 번만 계산 ---
    # (target/stop/max_hold 가 고정이면 결과는 gap 과 무관하므로, 구간별로 재계산할 필요 없음.
    #  슬라이딩 구간이 겹쳐 같은 pos 가 여러 구간에 들어가도 계산은 1회만 하고 인덱싱으로 집계.)
    exit_ret = np.full(n, np.nan)   # 청산 수익률(%)
    max_ret = np.full(n, np.nan)    # 보유 중 최대 도달 수익률(%)
    hold_day = np.zeros(n, dtype=np.int32)
    win = np.zeros(n, dtype=bool)
    # 200일선 복귀 방식 (200일선 아래 진입만): max_hold 내 종가가 200일선 위로 복귀하면 승리
    sma_win = np.zeros(n, dtype=bool)
    sma_has = np.zeros(n, dtype=bool)
    # 200일선 이탈 매도 방식: 종가 < 200일선×(1-완충) 이탈 시 청산, 그 시점 수익률 +면 승리
    br_win = np.zeros(n, dtype=bool)
    br_has = np.zeros(n, dtype=bool)
    BR_BUFFER = 0.05  # 이탈 완충 5% (콘텐츠 기준)

    # 모든 전략의 표본이 동일한 최대 보유기간을 갖도록 미래 데이터가
    # max_hold 거래일 온전히 남아 있는 진입일만 사용한다.
    for pos in range(max(0, n - max_hold)):
        entry = close[pos]
        end = pos + max_hold
        path = close[pos + 1:end + 1]
        cum = (path / entry - 1.0) * 100.0

        hit_t = np.argmax(cum >= target_pct) if (cum >= target_pct).any() else -1
        hit_s = np.argmax(cum <= -stop_pct) if (cum <= -stop_pct).any() else -1

        if hit_t != -1 and (hit_s == -1 or hit_t <= hit_s):
            exit_idx = int(hit_t)
            win[pos] = True
            exit_ret[pos] = target_pct
            hold_day[pos] = exit_idx + 1
        elif hit_s != -1:
            exit_idx = int(hit_s)
            exit_ret[pos] = -stop_pct
            hold_day[pos] = exit_idx + 1
        else:
            exit_idx = path.size - 1
            final = cum[exit_idx]
            win[pos] = final > 0
            exit_ret[pos] = final
            hold_day[pos] = path.size

        # 목표/손절/만기 중 실제 청산일까지의 최대 유리 수익률(MFE, 진입 시점 0% 포함).
        # 청산 뒤 반등·상승은 이 거래의 성과에 포함하지 않는다.
        max_ret[pos] = max(0.0, float(cum[:exit_idx + 1].max()))

        # 200일선 복귀: 진입 괴리율 -3% 이하 표본만 계산한다.
        if gap[pos] <= -3:
            fut_sma = sma[pos + 1:end + 1]
            sma_has[pos] = True
            sma_win[pos] = bool((path > fut_sma).any())

        # 200일선 이탈 매도: 진입 당시 이미 -5% 이탈선 아래인 표본은 제외한다.
        if gap[pos] > -BR_BUFFER * 100:
            fut_sma_b = sma[pos + 1:end + 1]
            sell_line = fut_sma_b * (1 - BR_BUFFER)
            breached = np.where(path < sell_line)[0]
            br_has[pos] = True
            if breached.size > 0:
                d = int(breached[0])
                br_win[pos] = bool(path[d] > entry)
            else:
                br_win[pos] = bool(path[-1] > entry)

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
        # 200일선 복귀 승률 (이 구간에서 200일선 아래 진입 표본이 있을 때만)
        sma_sel = sma_has & (gap >= lo) & (gap < hi)
        sma_n = int(sma_sel.sum())
        sma_wr = float(sma_win[sma_sel].mean() * 100) if sma_n > 0 else None
        # 200일선 이탈 매도 승률 (콘텐츠 방식)
        br_sel = br_has & (gap >= lo) & (gap < hi)
        br_n = int(br_sel.sum())
        br_wr = float(br_win[br_sel].mean() * 100) if br_n > 0 else None
        # 추세별 목표/손절 승률 (50일선>200일선=상승추세)
        up_sel = sel & uptrend
        down_sel = sel & (~uptrend)
        up_n = int(up_sel.sum())
        down_n = int(down_sel.sum())
        up_wr = float(win[up_sel].mean() * 100) if up_n > 0 else None
        down_wr = float(win[down_sel].mean() * 100) if down_n > 0 else None
        rows.append({
            "center": center,
            "zone_label": f"{fmt(lo)}%~{fmt(hi)}%",
            "trades": trades,
            "win_rate": float(win[sel].mean() * 100),
            "up_win_rate": up_wr, "up_trades": up_n,
            "down_win_rate": down_wr, "down_trades": down_n,
            "avg_return": float(exit_ret[sel].mean()),
            "max_return": float(np.nanmean(max_ret[sel])),
            "avg_holding_days": int(round(hold_day[sel].mean())),
            "sma_win_rate": sma_wr,
            "sma_trades": sma_n,
            "breach_win_rate": br_wr,
            "breach_trades": br_n,
        })

    return pd.DataFrame(rows)


def _match_zone_center(centers, gap: float, band_width: float):
    """Return the nearest populated center whose [lo, hi) interval contains gap."""
    half = float(band_width) / 2
    matches = [
        center for center in centers
        if float(center) - half <= gap < float(center) + half
    ]
    return min(matches, key=lambda center: abs(float(center) - gap)) if matches else None


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


def _rsi_wilder(close: pd.Series, period: int = 14):
    """종가 시리즈의 최신 RSI(Wilder EMA 방식). 값이 없으면 None."""
    if close is None or len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    last = rsi.iloc[-1]
    return float(last) if pd.notna(last) else None


def _reversal_signal(close: pd.Series, gap: float):
    """가격 데이터에서 관찰되는 '반전 조짐'을 종합 (예측 아님).

    상방 반전 조짐(바닥권): 과매도 + RSI 저점 반등 + 200일선 한참 아래 + 하락 다이버전스
    하방 반전 조짐(고점권): 과매수 + RSI 고점 꺾임 + 200일선 한참 위 + 상승 다이버전스
    Returns (라벨, 상세리스트) — 조짐 없으면 ('관찰되는 반전 조짐 없음', 근거).
    """
    if close is None or len(close) < 30:
        return "판단 불가 (데이터 부족)", []
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    al = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = (100 - 100 / (1 + ag / al)).dropna()
    if len(rsi) < 15:
        return "판단 불가 (데이터 부족)", []

    cur_rsi = float(rsi.iloc[-1])
    recent = rsi.tail(10)
    rsi_min, rsi_max = float(recent.min()), float(recent.max())
    px = close.dropna().tail(10)

    up_pts, down_pts = [], []  # 상방 조짐 / 하방 조짐 근거
    # 1) 과매도/과매수
    if cur_rsi <= 30:
        up_pts.append(f"RSI {cur_rsi:.0f} 과매도")
    elif cur_rsi >= 70:
        down_pts.append(f"RSI {cur_rsi:.0f} 과매수")
    # 2) RSI가 최근 저점/고점에서 돌아섰는지
    if cur_rsi <= 40 and cur_rsi >= rsi_min + 3:
        up_pts.append("RSI 저점에서 반등 중")
    if cur_rsi >= 60 and cur_rsi <= rsi_max - 3:
        down_pts.append("RSI 고점에서 꺾임")
    # 3) 다이버전스: 가격 신저점인데 RSI는 더 높음 / 가격 신고점인데 RSI는 더 낮음
    if len(px) >= 6 and float(px.iloc[-1]) <= float(px.iloc[:-1].min()) and cur_rsi > rsi_min + 2:
        up_pts.append("가격 신저점 + RSI 상승(강세 다이버전스)")
    if len(px) >= 6 and float(px.iloc[-1]) >= float(px.iloc[:-1].max()) and cur_rsi < rsi_max - 2:
        down_pts.append("가격 신고점 + RSI 하락(약세 다이버전스)")
    # 4) 200일선에서 극단적으로 벌어짐 (평균회귀 압력)
    if gap <= -20:
        up_pts.append(f"200일선 {gap:+.0f}%로 과도한 하락")
    elif gap >= 25:
        down_pts.append(f"200일선 {gap:+.0f}%로 과도한 상승")

    if len(up_pts) >= 2 and len(up_pts) > len(down_pts):
        return "🟢 상방 반전 조짐 (바닥권 신호 우세)", up_pts
    if len(down_pts) >= 2 and len(down_pts) > len(up_pts):
        return "🔴 하방 반전 조짐 (고점권 신호 우세)", down_pts
    if up_pts and len(up_pts) >= len(down_pts):
        return "🟡 약한 상방 조짐 (근거 부족)", up_pts
    if down_pts:
        return "🟡 약한 하방 조짐 (근거 부족)", down_pts
    return "⚪ 뚜렷한 반전 조짐 없음", []


def _mvrv_zone(v):
    if v is None or not np.isfinite(v):
        return None
    for zone, (lo, hi) in _MVRV_ZONES.items():
        if lo <= v < hi:
            return zone
    return None


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
            rsi = _rsi_wilder(d["Close"])
            d = d.dropna()
            price = float(d["Close"].iloc[-1])
            sma = float(d["SMA200"].iloc[-1])
            gap = (price / sma - 1) * 100
            above = price > sma
            sell_line = sma * (1 - asset['buffer'])
            rows.append({
                "자산": asset['short'], "현재가": price, "200일선": sma,
                "괴리율": gap, "위/아래": "위" if above else "아래",
                "매도선": sell_line, "이탈": price < sell_line, "rsi": rsi,
            })

    if not rows:
        st.error("크립토 데이터를 불러오지 못했어요.")
        return

    # MVRV 요약
    if btc_mvrv is not None and len(btc_mvrv) > 0:
        mv = float(btc_mvrv.iloc[-1]["mvrv"])
        zone = _mvrv_zone(mv)
        mv_date = btc_mvrv.index[-1].strftime("%Y-%m-%d")
        if zone is None:
            st.warning(f"BTC MVRV 최신값이 유효하지 않아요 (기준일 {mv_date}). 가격/200일선만 참고하세요.")
        else:
            emoji, label = _MVRV_LABEL[zone]
            st.markdown(f"**BTC MVRV**: {mv:.3f} {emoji} — {label}  \n"
                        f"<span style='color:gray'>기준일 {mv_date} · MVRV 1.5 이하면 크립토 전반 저평가</span>",
                        unsafe_allow_html=True)
    else:
        st.warning("BTC MVRV 데이터를 불러오지 못했어요 (외부 API). 가격/200일선 정보만 표시합니다.")

    # 대시보드 테이블
    table = pd.DataFrame([{
        "자산": r["자산"],
        "현재가": f"{r['현재가']:,.2f}",
        "괴리율": f"{r['괴리율']:+.1f}%",
        "위/아래": r["위/아래"],
        "RSI": _fmt_rsi(r.get("rsi")),
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
        if ad <= 2:
            score = 10
        elif ad >= 15:
            score = 100
        else:
            # 2%p의 10점에서 15%p의 100점까지 단조롭게 증가한다.
            score = int(round(10 + (ad - 2) / 13 * 90))
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

    # 하이일드 신용 스프레드 (FRED, '스마트 머니'가 가장 먼저 반응)
    hy = _hy_spread()
    if hy is not None:
        bp = hy * 100  # % -> bp
        if bp >= 800: score, stt = 100, "점등"
        elif bp >= 500: score, stt = 80, "점등"
        elif bp >= 450: score, stt = 55, "경계"
        elif bp >= 400: score, stt = 35, "경계"
        else: score, stt = 15, "정상"
        out.append(("하이일드 스프레드", f"{bp:.0f}bp", stt, score,
                    "500bp+ 확대 = 신용시장 공포, 주식보다 2~4주 선행"))

    # 소비자 신뢰지수 (FRED UMCSENT, 침체 선행)
    conf = _consumer_confidence()
    if conf is not None:
        if conf < 55: score, stt = 85, "점등"
        elif conf < 70: score, stt = 50, "경계"
        elif conf < 90: score, stt = 25, "정상"
        else: score, stt = 10, "정상"
        out.append(("소비자 신뢰지수 (미시건대)", f"{conf:.1f}", stt, score,
                    "55↓ = 역사적 저점(침체 선행). 소비=GDP 70%"))

    return out


@st.cache_data(ttl=3600, show_spinner=False, max_entries=1)
def _hy_spread():
    """FRED 하이일드 스프레드(%) 최근값. 실패 시 None."""
    try:
        import pandas_datareader.data as web
        from datetime import datetime as _dt, timedelta as _td
        df = web.DataReader("BAMLH0A0HYM2", "fred",
                            _dt.now() - _td(days=30), _dt.now())
        df = df.dropna()
        return float(df.iloc[-1, 0]) if len(df) else None
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False, max_entries=1)
def _consumer_confidence():
    """FRED 미시건대 소비자심리지수(UMCSENT) 최근값. 실패 시 None. (월간 데이터)"""
    try:
        import pandas_datareader.data as web
        from datetime import datetime as _dt, timedelta as _td
        df = web.DataReader("UMCSENT", "fred",
                            _dt.now() - _td(days=200), _dt.now())
        df = df.dropna()
        return float(df.iloc[-1, 0]) if len(df) else None
    except Exception:
        return None


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

    box(f"**종합 위험도 {overall}/100 [{level}]** · 자동 지표 {len(inds)}/6개 로드 "
        f"(점등 {lit} / 경계 {caution} / 정상 {normal})  \n권고: {rec}")

    mark = {"점등": "🔴", "경계": "🟡", "정상": "🟢"}
    st.markdown("#### 📡 자동 점등 지표 (실시간)")
    table = pd.DataFrame([{
        "지표": name, "값": val, "상태": f"{mark.get(stt,'⚪')} {stt}",
        "점수": score, "설명": desc,
    } for name, val, stt, score, desc in inds])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("⚠️ 일부 매크로 지표는 시장 데이터 기반 근사치입니다. 참고용이며 투자 권유가 아닙니다.")

    # --- 확인 체크리스트 (자동화 어려운 매크로 지표: 링크+기준) ---
    with st.expander("📋 추가 확인 지표 체크리스트 (수동 확인)", expanded=False):
        st.markdown("""
자동 계산이 어려운 매크로 지표들이에요. 아래 링크에서 직접 확인하세요.
과거 대형 폭락(2000·2008·2022) 전엔 이런 지표가 **9~10개 동시 점등**됐어요.

| 지표 | 점등 기준 | 확인 |
|---|---|---|
| **버핏 지표** (시총/GDP) | 200%↑ 위험 (장기평균 88%) | [gurufocus](https://www.gurufocus.com/stock-market-valuations.php) |
| **쉴러 CAPE** | 30↑ 고평가, 40↑ 닷컴급 | [multpl](https://www.multpl.com/shiller-pe) |
| **내부자 매도** | 여러 기업 동시 대량 매도(클러스터) | [openinsider](http://openinsider.com) |
| **ISM 제조업 PMI** | 50 하회 (특히 45↓) | [ISM](https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/) |
| **마진 부채** | 사상 최고 후 감소 전환 | [FINRA](https://www.finra.org/investors/learn-to-invest/advanced-investing/margin-statistics) |
| **연준 사이클** | 마지막 인상 후 6~18개월 (최고 위험) | [FRED FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) |
| **"이번엔 다르다" 내러티브** | 미디어·개인 낙관 만연, IPO 열풍 | 정성적 판단 |
        """)

    # --- 3단계 대응 매뉴얼 ---
    with st.expander("🛡️ 경고 점등 시 대응 매뉴얼", expanded=False):
        st.markdown("""
**점등 개수별 3단계 대응**

| 단계 | 점등 수 | 행동 |
|---|---|---|
| 1단계 (관찰) | 3~4개 | 신규 매수 속도 ↓, 현금 5~10% 확보 |
| 2단계 (경계) | 5~7개 | 레버리지 해제, 현금 20~30%, 방어주 확대 |
| 3단계 (방어) | 8개↑ | 주식 50%↓, 장기국채(TLT)·금(GLD) 편입 |

**절대 금지**: 공포에 전량 매도 · 레버리지 추가 · 한 종목 올인
**반드시**: 현금 확보 · 분할매수 계획(-10/-20/-30%) · 손절 라인 사전 설정

**⏱️ 이것만 보면 되는 3가지** (동시 점등 시 진짜 하락장 확률↑)
1. **하이일드 스프레드** 급등 (스마트머니 2~4주 선행)
2. **S&P500 200일선 이탈** (추세 붕괴)
3. **VIX 30 돌파** (패닉 진입 = 역발상 매수 대기)

<span style='color:gray'>· 구조적 폭락은 미리 보이지만(9~10개 점등), 외부충격(코로나 등)은 예측 불가.
· 지표 하나로 판단 말고 '합류'를 보세요. 타이밍보다 비중 조절이 정답.
· 과거 데이터 기반이며 투자 권유가 아닙니다.</span>
        """, unsafe_allow_html=True)


# ============================================================
# 추가 도구 3: 데일리 스캐너 (200일선 상태 모니터)
# ============================================================
_DS_INDICES = {'QQQ': '나스닥100', 'SPY': 'S&P500', '069500.KS': '코스피(ETF)'}
_DS_US_TIER1 = ['NVDA', 'META', 'GOOGL', 'AAPL', 'CRM', 'MA', 'CSCO']

# 주요 지수·레버리지 ETF 200일선 스캔 목록 (티커: 이름)
_DS_LEV_INDICES = {
    '^KS11': '코스피 지수', '^KQ11': '코스닥 지수',
    '^IXIC': '나스닥 종합', '^GSPC': 'S&P500 지수',
    'QQQ': '나스닥100(QQQ)', 'SOXX': '반도체(SOXX)',
    'TQQQ': 'TQQQ(나스닥100 3x)', 'SOXL': 'SOXL(반도체 3x)',
    'KORU': 'KORU(한국 3x)', 'GDXU': 'GDXU(금광 2x)',
    'BTC-USD': 'BTC(비트코인)', 'ETH-USD': 'ETH(이더리움)',
}

# 환율·국채 200일선 스캔 목록 (티커: 이름)
_DS_FX_BOND = {
    'KRW=X': '달러/원 (USDKRW)', 'JPY=X': '달러/엔 (USDJPY)',
    'DX-Y.NYB': '달러인덱스 (DXY)',
    'SHY': '미국채 1-3년 (SHY)', 'IEF': '미국채 7-10년 (IEF)',
    'TLT': '미국채 20년+ (TLT)',
    '^TNX': '미국채 10년 금리', '^TYX': '미국채 30년 금리',
}
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
    rsi = _rsi_wilder(data["Close"])

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
    return {"close": close, "ma": ma, "gap": gap, "above": close > ma,
            "signal": signal, "rsi": rsi}


def detect_box(df: pd.DataFrame, lookback: int = 60, tol: float = 0.03):
    """
    최근 lookback일 가격으로 박스권(횡보 구간)을 자동 탐지.
    박스권 판정: 최근 구간의 (고점-저점)/중앙값 범위가 좁고, 종가가 그 안에서 횡보.

    Returns dict:
      is_box: 박스권 여부
      top, bottom: 박스 상단/하단
      width_pct: 박스 폭 (%)
      pos_pct: 현재가의 박스 내 위치 (0=하단, 100=상단)
      status: 상태 문자열
    """
    if df is None or len(df) < lookback + 1:
        return None
    # 현재 봉은 박스 기준 고저점에서 제외해야 돌파/이탈을 판정할 수 있다.
    close = df["Close"].values[-(lookback + 1):-1]
    cur = float(df["Close"].iloc[-1])

    hi = float(np.max(close))
    lo = float(np.min(close))
    mid = (hi + lo) / 2
    if mid <= 0:
        return None
    width_pct = (hi - lo) / mid * 100

    # 박스권 조건: 최근 구간 폭이 좁음(예: 25% 이내) + 추세가 뚜렷하지 않음
    # 추세 판정: 구간을 반으로 나눠 평균 차이가 작으면 횡보
    half = lookback // 2
    first_avg = float(np.mean(close[:half]))
    second_avg = float(np.mean(close[half:]))
    trend_pct = abs(second_avg - first_avg) / mid * 100

    is_box = width_pct <= 25 and trend_pct <= 8

    # 현재가의 박스 내 위치 (0~100)
    pos_pct = (cur - lo) / (hi - lo) * 100 if hi > lo else 50

    # 상태 판정
    if cur > hi * (1 + tol):
        status = "🟢 박스 상단 돌파 (매수 신호 가능)"
    elif cur < lo * (1 - tol):
        status = "🔴 박스 하단 이탈 (매도/이탈)"
    elif pos_pct >= 80:
        status = "🔵 박스 상단 근처 (돌파 대기)"
    elif pos_pct <= 20:
        status = "🟡 박스 하단 근처 (지지 테스트)"
    else:
        status = "⚪ 박스 중간"

    return {
        "is_box": is_box,
        "top": hi, "bottom": lo, "mid": mid,
        "width_pct": width_pct, "trend_pct": trend_pct,
        "pos_pct": pos_pct, "cur": cur, "status": status,
        "lookback": lookback,
    }


@st.cache_data(ttl=3600, show_spinner=False, max_entries=60)
def get_fundamentals(ticker):
    """단일 종목의 밸류에이션+재무건전성 지표를 yfinance .info에서 실시간 조회.
    Returns dict 또는 None. 코인/지수 등 지표 없으면 대부분 None 필드."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None

    def num(*keys):
        for k in keys:
            v = info.get(k)
            if isinstance(v, (int, float)) and v == v:  # not NaN
                return float(v)
        return None

    per = num("trailingPE", "forwardPE")
    # 국내 종목은 trailingPE가 비고 forwardPE만 오는 경우가 많아 구분해 표시한다.
    per_forward = per is not None and not isinstance(info.get("trailingPE"), (int, float))
    pbr = num("priceToBook")
    psr = num("priceToSalesTrailing12Months")
    roe = num("returnOnEquity")            # 소수 (0.15 = 15%)
    dte = num("debtToEquity")              # % 또는 배수 (yfinance는 보통 % 단위, 예: 150 = 150%)
    opm = num("operatingMargins")          # 소수 (0.25 = 25%)
    return {"per": per, "pbr": pbr, "psr": psr, "roe": roe, "dte": dte, "opm": opm,
            "per_forward": per_forward}


def _fmt_valuation(f):
    """밸류에이션 판단 (PER/PBR/PSR 절대 기준). (요약라벨, 상세리스트) 반환."""
    if not f:
        return "-", []
    details = []
    flags = []  # 고평가 신호 개수
    per, pbr, psr = f.get("per"), f.get("pbr"), f.get("psr")
    if per is not None:
        if per <= 0:
            tag = "⚪적자/의미 없음"
        else:
            tag = "🔴높음" if per >= 30 else ("🟡보통" if per >= 15 else "🟢낮음")
            if per >= 30:
                flags.append(1)
        details.append(("PER(예상)" if f.get("per_forward") else "PER", f"{per:.1f}", tag))
    if pbr is not None:
        tag = "🔴높음" if pbr >= 5 else ("🟡보통" if pbr >= 1.5 else "🟢낮음")
        if pbr >= 5:
            flags.append(1)
        details.append(("PBR", f"{pbr:.2f}", tag))
    if psr is not None:
        tag = "🔴높음" if psr >= 10 else ("🟡보통" if psr >= 3 else "🟢낮음")
        if psr >= 10:
            flags.append(1)
        details.append(("PSR", f"{psr:.2f}", tag))
    if not details:
        return "-", []
    meaningful = int(per is not None and per > 0) + int(pbr is not None and pbr > 0) + int(psr is not None and psr > 0)
    if meaningful == 0:
        summary = "⚪ 판단 유보"
    elif len(flags) >= 2:
        summary = "🔴 고평가 경향"
    elif len(flags) == 1:
        summary = "🟡 일부 고평가"
    else:
        summary = "🟢 부담 적음"
    return summary, details


def _fmt_health(f):
    """재무건전성 판단 (ROE/부채비율/영업이익률). (요약라벨, 상세리스트) 반환."""
    if not f:
        return "-", []
    details = []
    good = 0
    total = 0
    roe, dte, opm = f.get("roe"), f.get("dte"), f.get("opm")
    if roe is not None:
        total += 1
        pct = roe * 100
        tag = "🟢우수" if pct >= 15 else ("🟡보통" if pct >= 5 else "🔴낮음")
        if pct >= 15:
            good += 1
        details.append(("ROE", f"{pct:.1f}%", tag))
    if dte is not None:
        total += 1
        # yfinance debtToEquity는 % 단위(예: 150 = 부채/자본 150%)
        tag = "🟢낮음" if dte < 100 else ("🟡보통" if dte < 200 else "🔴높음")
        if dte < 100:
            good += 1
        details.append(("부채비율", f"{dte:.0f}%", tag))
    if opm is not None:
        total += 1
        pct = opm * 100
        tag = "🟢우수" if pct >= 20 else ("🟡보통" if pct >= 8 else "🔴낮음")
        if pct >= 20:
            good += 1
        details.append(("영업이익률", f"{pct:.1f}%", tag))
    if total == 0:
        return "-", []
    if good >= 2:
        summary = "🟢 건전"
    elif good == 1:
        summary = "🟡 보통"
    else:
        summary = "🔴 주의"
    return summary, details


def _fmt_rsi(rsi):
    """RSI 값을 상태 라벨과 함께 문자열로. (과매도<30, 과매수>70)"""
    if rsi is None:
        return "-"
    if rsi >= 70:
        return f"{rsi:.0f} 🔴과매수"
    if rsi <= 30:
        return f"{rsi:.0f} 🟢과매도"
    return f"{rsi:.0f}"


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
            "종목": name, "종가": f"{r['close']:,.2f}",
            "괴리율": f"{r['gap']:+.1f}%", "위/아래": "위" if r["above"] else "아래",
            "RSI": _fmt_rsi(r.get("rsi")),
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


def _winzone_lookup(ticker, gap, mode="target"):
    """winzone_data에서 현재 gap에 맞는 구간 승률 조회.
    mode: 'target'=목표+10/손절-5, 'sma'=200일선 복귀 매도.
    Returns (현재구간승률str, 최고승률구간str) — 데이터 없으면 ('-','-')."""
    v = WINZONE_DATA.get(ticker)
    if not v:
        return "-", "-"
    # 200일선 복귀 모드는 200일선 아래(-3% 이하)에서만 의미 있음
    if mode == "sma" and gap > -3:
        return "-", "-"
    key = "zones_sma" if mode == "sma" else "zones"
    zones = v.get(key)
    if not zones:
        return "-", "-"
    # 현재 gap을 실제로 포함하는 사전계산 구간만 매칭한다.
    band = float(WINZONE_META.get("band", 10))
    best_center = _match_zone_center(zones.keys(), gap, band)
    if best_center is None:
        cur_str = "-"
    else:
        wr, samp = zones[best_center]
        cur_str = f"{wr:.0f}% ({samp}건)"
    # 최고 승률 구간
    bz = max(zones.items(), key=lambda kv: kv[1][0])
    best_str = f"{int(bz[0]):+d}% ({bz[1][0]:.0f}%)"
    return cur_str, best_str


def render_kr_winzone():
    st.markdown("#### 🇰🇷 국장 대형주 200일선 매수 전략 스캐너")
    st.caption("'200일선 아래 -N%에서 매수 → 200일선 복귀 시 매도' 전략. "
               "현재 위치와 그 구간의 역사적 승률을 함께 봅니다.  \n"
               "📏 **승률 기준: 200일선 복귀 시 매도** (전수조사 콘텐츠 값). "
               "⚠️ '승률 포착기'·미국/알트 스캐너는 **+10% 목표 선도달 또는 3개월 만기 양수**를 성공으로 "
               "보는 다른 기준이라 같은 종목도 값이 달라요.")

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
                "RSI": _fmt_rsi(r.get("rsi")),
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
                "RSI": _fmt_rsi(r.get("rsi")),
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
        "역사적으로 가장 높았던 매수 승률인 앱 내 참고값입니다. "
        "이 저장소에는 해당 국장 통계의 생성 산식·기준일이 없어 독립 재현 검증하지 못했습니다. "
        "200일선 위 매수의 승률이 아니라, '내려오면 여기가 기회'라는 참고 지표예요.</span>",
        unsafe_allow_html=True)


def render_us_winzone():
    st.markdown("#### 🇺🇸 미국 대형주 200일선 매수 전략 스캐너")
    st.caption("'200일선 아래로 내려오면 매수' 전략. "
               "복귀 빠른 TOP50 종목의 현재 위치·구간 승률·복귀 기간을 함께 봅니다.  \n"
               "📏 **성공 기준: +10% 목표 선도달 또는 3개월 만기 양수 / -5% 손절 선도달은 실패** "
               "(국장 스캐너의 '200일선 복귀 시 매도' 기준과 달라요).")

    # 복귀 빠른 순(평균 복귀일)으로 정렬, 중복 티커 제거
    seen = set()
    assets = []
    for tk, name, cyc, avg, med, mx in _RECOVERY_TOP50:
        if tk in seen:
            continue
        seen.add(tk)
        assets.append((tk, name, avg, med, mx))
    assets.sort(key=lambda x: x[2])  # 평균 복귀일 오름차순

    below_rows = []   # 지금 200일선 아래 (매수 기회)
    above_rows = []   # 200일선 위 (대기)
    for tk, name, avg, med, mx in assets:
        r = _ds_status(tk)
        if not r:
            continue
        gap = r["gap"]
        recov = f"평균 {avg}일 / 중앙값 {med}일"
        # 복귀 기간 기반 대응 가이드 (평균 = 깊게 빠졌을 때 모을 수 있는 시간)
        guide = _recovery_action(avg)

        cur_wr, best_wr = _winzone_lookup(tk, gap)

        if gap < 0:
            # 200일선 아래 = 매수 기회
            if gap <= -10:
                status = "🟢 깊은 매수 구간 (-10% 이하)"
            elif gap <= -5:
                status = "🟢 매수 구간 (-5% 이하)"
            else:
                status = "⚪ 200일선 바로 아래 · 매수 임박"
            below_rows.append({
                "종목": name,
                "현재 괴리율": f"{gap:+.1f}%",
                "RSI": _fmt_rsi(r.get("rsi")),
                "구간 승률": cur_wr,
                "최고 승률 구간": best_wr,
                "복귀 기간": recov,
                "대응": guide,
                "상태": status,
            })
        else:
            above_rows.append({
                "종목": name,
                "현재 괴리율": f"{gap:+.1f}%",
                "RSI": _fmt_rsi(r.get("rsi")),
                "구간 승률": cur_wr,
                "최고 승률 구간": best_wr,
                "복귀 기간": recov,
                "상태": f"🔵 200일선 위 (+{gap:.1f}%) · 대기",
            })

    if not below_rows and not above_rows:
        st.warning("데이터를 불러오지 못했어요.")
        return

    # 1) 지금 200일선 아래 (매수 기회) — 깊이 빠진 순으로
    if below_rows:
        st.success(f"🎯 지금 200일선 아래로 내려온 종목: **{len(below_rows)}개** (매수 기회)")
        below_df = pd.DataFrame(below_rows)
        below_df["_sort"] = below_df["현재 괴리율"].str.rstrip("%").astype(float)
        below_df = below_df.sort_values("_sort").drop(columns="_sort")
        st.dataframe(below_df, use_container_width=True, hide_index=True)
    else:
        st.info("현재 200일선 아래로 내려온 종목이 없어요. 대부분 200일선 위입니다.")

    # 2) 200일선 위 (대기) — 200일선에 가까운 순으로
    if above_rows:
        st.markdown("**📈 200일선 위 / 대기 종목** — 지금은 매수 구간 아님. "
                    "'복귀 기간'은 이 종목이 200일선 아래로 내려갔을 때 역사적으로 며칠 만에 복귀했는지예요.")
        above_df = pd.DataFrame(above_rows)
        above_df["_sort"] = above_df["현재 괴리율"].str.rstrip("%").astype(float)
        above_df = above_df.sort_values("_sort").drop(columns="_sort")
        st.dataframe(above_df, use_container_width=True, hide_index=True)

    st.markdown(
        "<span style='color:gray'>· 복귀 빠른 순(평균 복귀일)으로 정렬. "
        "복귀가 빠른 종목일수록 200일선 아래에서 모을 시간이 짧으니 '내려오자마자' 사야 하고, "
        "느린 종목은 여유 있게 분할 매수할 수 있어요. "
        "복귀 기간은 미국 시총 TOP50 전수조사(7,900 사이클, 미복귀 구간 제외, 달력일 기준) 내장값이에요. "
        "중앙값은 대부분 4~7일로 비슷하므로 '모을 시간'은 평균 복귀일로 판단합니다.</span>",
        unsafe_allow_html=True)


def render_alt_winzone():
    st.markdown("#### 🪙 알트코인 200일선 매수 전략 스캐너")
    st.caption("주요 알트코인의 현재 200일선 위치와 그 위치의 역사적 승률을 봅니다.  \n"
               "📏 **성공 기준: +10% 목표 선도달 또는 3개월 만기 양수 / -5% 손절 선도달은 실패** "
               "(국장 스캐너의 '200일선 복귀 시 매도' 기준과 달라요).")

    alts = [(tk, v["name"]) for tk, v in WINZONE_DATA.items() if v.get("market") == "ALT"]
    if not alts:
        st.info("알트코인 승률 데이터가 없어요.")
        return

    rows = []
    for tk, name in alts:
        r = _ds_status(tk)
        if not r:
            continue
        gap = r["gap"]
        cur_wr, best_wr = _winzone_lookup(tk, gap)
        if gap < 0:
            status = "🟢 매수 구간 (200일선 아래)" if gap <= -5 else "⚪ 200일선 바로 아래"
        else:
            status = f"🔵 200일선 위 (+{gap:.1f}%) · 대기"
        rows.append({
            "코인": name,
            "현재 괴리율": f"{gap:+.1f}%",
            "RSI": _fmt_rsi(r.get("rsi")),
            "구간 승률": cur_wr,
            "최고 승률 구간": best_wr,
            "상태": status,
        })

    if not rows:
        st.warning("데이터를 불러오지 못했어요.")
        return
    df = pd.DataFrame(rows)
    df["_sort"] = df["현재 괴리율"].str.rstrip("%").astype(float)
    df = df.sort_values("_sort").drop(columns="_sort")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("· '구간 성공률'은 현재 괴리율을 실제로 포함하는 사전계산 구간의 성공률(표본수)이에요. "
               "포함 구간이 없으면 '-'로 표시합니다. "
               "· 코인은 변동성이 커서 표본·성공률 해석에 주의하세요. 과거 성과가 미래를 보장하지 않습니다.")


def render_fxb_winzone():
    st.markdown("#### 💱 환율 · 국채 200일선 승률 스캐너")
    st.caption("환율·국채의 현재 200일선 위치와 목표/손절 전략의 역사적 성공률을 봅니다.  \n"
               "⚠️ 성공률은 +10% 목표 선도달뿐 아니라, 목표·손절 미도달 시 3개월 만기 수익이 +인 경우도 포함합니다. "
               "환율·국채는 주식과 성격이 다르므로 같은 기준으로 직접 비교하지 마세요.")

    fxbs = [(tk, v["name"]) for tk, v in WINZONE_DATA.items() if v.get("market") == "FXB"]
    if not fxbs:
        st.info("환율·국채 승률 데이터가 없어요.")
        return

    rows = []
    for tk, name in fxbs:
        r = _ds_status(tk)
        if not r:
            continue
        gap = r["gap"]
        cur_wr, best_wr = _winzone_lookup(tk, gap)
        status = "🔵 200일선 위" if gap >= 0 else "🟢 200일선 아래"
        rows.append({
            "종목": name,
            "현재 괴리율": f"{gap:+.1f}%",
            "RSI": _fmt_rsi(r.get("rsi")),
            "구간 승률": cur_wr,
            "최고 승률 구간": best_wr,
            "상태": status,
        })

    if not rows:
        st.warning("데이터를 불러오지 못했어요.")
        return
    df = pd.DataFrame(rows)
    df["_sort"] = df["현재 괴리율"].str.rstrip("%").astype(float)
    df = df.sort_values("_sort").drop(columns="_sort")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("· 표본이 적은 구간(달러인덱스·단기채 등)은 신뢰도가 낮을 수 있어요. "
               "과거 성과가 미래를 보장하지 않습니다.")


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
            rows.append({"종목": name, "종가": "-",
                         "괴리율": "-", "위/아래": "-", "RSI": "-", "신호": "데이터 없음"})
            continue
        rows.append({
            "종목": name, "종가": f"{r['close']:,.2f}",
            "괴리율": f"{r['gap']:+.1f}%", "위/아래": "위" if r["above"] else "아래",
            "RSI": _fmt_rsi(r.get("rsi")),
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
    ("TXN", "TXN", 237, 33, 6, 463),
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


def _recovery_stats(ticker):
    """티커의 200일선 복귀 통계 (평균, 중앙값, 최대) 반환. 없으면 None.
    _RECOVERY_TOP50은 접미사 없는 미국 시총 TOP50 티커만 담고 있다."""
    key = str(ticker).upper().replace(".KS", "").replace(".KQ", "")
    for tk, _name, _cyc, avg, med, mx in _RECOVERY_TOP50:
        if tk.upper() == key:
            return {"avg": avg, "median": med, "max": mx}
    return None


def _recovery_action(avg_days):
    """평균 복귀일 기준 행동 가이드.
    중앙값은 모든 종목이 4~7일로 거의 같으므로, '모을 시간'을 결정하는 건
    깊게 빠졌을 때의 체류 기간(평균)이다. 원문 전수조사도 평균으로 구분한다."""
    if avg_days <= 25:
        return "복귀 빠름 → 내려오자마자 바로 매수 (모을 시간 짧음)"
    elif avg_days <= 40:
        return "복귀 보통 → 기본 분할 매수"
    else:
        return "복귀 느림 → 여유 있게 분할 매수 가능 (모을 시간 충분)"


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


def _current_gap(ticker):
    """현재 200일선 대비 괴리율(%) 반환. 실패 시 None."""
    r = _ds_status(ticker)
    return r["gap"] if r else None


# 시장별 대표 지수 (강세장 판단용)
_MARKET_INDEX = {
    "KR": ["^KS11", "^KQ11"],        # 국장: 코스피/코스닥
    "US": ["^GSPC", "^IXIC"],        # 미국: S&P500/나스닥
}


def _market_is_bull(market, min_wr=60):
    """해당 시장이 강세장인지: 대표 지수가 200일선 위 + 현재 위치 승률 min_wr%↑.
    Returns (강세여부, 설명리스트)."""
    idxs = _MARKET_INDEX.get(market, [])
    if not idxs:
        return True, []  # 판단 지수 없으면 통과 (알트/환율 등)
    details = []
    bull_any = False
    for ix in idxs:
        v = WINZONE_DATA.get(ix)
        r = _ds_status(ix)
        if not v or not r:
            continue
        gap = r["gap"]
        cur_wr, _ = _winzone_lookup(ix, gap, "target")
        wr_num = None
        if cur_wr != "-":
            try:
                wr_num = float(cur_wr.split("%")[0])
            except Exception:
                wr_num = None
        is_bull = r["above"] and (wr_num is not None and wr_num >= min_wr)
        if is_bull:
            bull_any = True
        details.append(f"{v['name']}: {gap:+.1f}% ({'위' if r['above'] else '아래'}), 승률 {cur_wr}")
    return bull_any, details


def render_winzone_catcher():
    st.subheader("🎯 승률 포착기")
    st.caption("미장·국장 대형주 200종목 중, 지금 위치가 역사적으로 고승률이었던 종목을 찾아줍니다.")

    if not WINZONE_DATA:
        st.error("사전 계산 데이터(winzone_data.py)를 찾을 수 없어요.")
        return

    meta = WINZONE_META
    generated_date = str(meta.get("generated_at", "미기록"))[:10]
    valuation_date = str(meta.get("valuation_as_of", "미기록"))[:10]
    sell_mode = st.radio(
        "매도 기준",
        ["목표 +10% / 손절 -5%", "200일선 복귀 시 매도"],
        horizontal=True, key="winzone_sellmode",
        help="승률을 어떤 매도 전략 기준으로 볼지 선택하세요.")
    mode = "sma" if sell_mode.startswith("200일선") else "target"

    if mode == "target":
        st.info(f"📌 **전략 성공률 정의**: 목표 **+{meta['target']:.0f}%** 먼저 도달은 성공, "
                f"손절 **-{meta['stop']:.0f}%** 먼저 도달은 실패. 둘 다 미도달하면 "
                f"**{meta['max_hold']}거래일(약 3개월) 만기 수익이 +면 성공**입니다. "
                f"완전한 보유기간이 남은 과거 진입만 집계합니다.")
    else:
        st.info(f"📌 **복귀 성공률 정의**: 괴리율 **-3% 이하**에서 매수 → "
                f"최대 {meta['max_hold']}거래일 내 종가가 200일선 위로 복귀하면 성공. "
                f"완전한 보유기간이 남은 과거 진입만 집계합니다.")

    c1, c2 = st.columns([1, 1])
    with c1:
        threshold = st.selectbox("승률 임계값", [59, 60, 70, 80, 90, 100], index=3,
                                 format_func=lambda x: f"{x}% 이상",
                                 help="현재 위치의 역사적 승률이 이 값 이상인 종목만 표시")
    with c2:
        market_filter = st.selectbox("시장", ["전체", "미국", "국장", "알트", "환율/국채"], index=0)

    bull_only = st.checkbox(
        "🐂 강세장 필터 (지수 200일선 위 + 지수 성공률 60%↑ 시장의 200일선 위 종목만)",
        value=False, key="winzone_bull", disabled=(mode == "sma"),
        help="목표/손절 모드에서만 사용합니다. 시장 지수가 강세일 때 그 시장의 200일선 위 종목만 봅니다.")
    if mode == "sma":
        bull_only = False
        st.caption("· 200선복귀 모드는 -3% 이하 진입만 보므로, 200일선 위만 고르는 강세장 필터는 적용하지 않습니다.")

    if not st.button("🔍 승률 포착 스캔", type="primary", key="winzone_scan"):
        st.info("버튼을 눌러 지금 고승률 구간에 있는 종목을 찾아보세요. "
                f"(200종목+ 현재가 확인, 20~40초 소요)")
        return

    mkmap = {"미국": "US", "국장": "KR", "알트": "ALT", "환율/국채": "FXB"}
    # 지수(IDX)는 종목이 아니라 시장 판단용이므로 스캔 대상에서 제외
    targets = [(tk, v) for tk, v in WINZONE_DATA.items()
               if v["market"] != "IDX"
               and (market_filter == "전체" or v["market"] == mkmap.get(market_filter))]

    # 강세장 필터: 각 시장의 강세 여부를 미리 판정
    bull_status = {}
    if bull_only:
        for mk in ("US", "KR"):
            is_bull, details = _market_is_bull(mk, min_wr=60)
            bull_status[mk] = is_bull
        # 강세장 상태 안내
        kr_txt = "🟢 강세" if bull_status.get("KR") else "🔴 약세"
        us_txt = "🟢 강세" if bull_status.get("US") else "🔴 약세"
        st.markdown(f"**시장 상태** — 국장(코스피/코스닥): {kr_txt} · 미국(S&P/나스닥): {us_txt}")

    zones_key = "zones_sma" if mode == "sma" else "zones"
    hits = []
    prog = st.progress(0.0)
    total = len(targets)
    for i, (tk, v) in enumerate(targets):
        prog.progress((i + 1) / total)
        r = _ds_status(tk)
        if not r:
            continue
        gap = r["gap"]
        # 강세장 필터: 시장이 강세이고, 종목이 200일선 위일 때만
        if bull_only:
            if not bull_status.get(v["market"], False):
                continue  # 이 종목 시장이 약세면 제외
            if gap < 0:
                continue  # 200일선 아래면 제외 (강세장 = 위 종목만)
        # 200일선 복귀 모드는 200일선 아래(-3% 이하)에서만 의미 있음
        if mode == "sma" and gap > -3:
            continue
        # 현재 gap을 실제로 포함하는 사전계산 구간만 매칭한다.
        zones = v.get(zones_key)
        if not zones:
            continue
        best_center = _match_zone_center(zones.keys(), gap, float(meta.get("band", 10)))
        if best_center is None:
            continue
        wr, samples = zones[best_center]
        if wr >= threshold:
            mk_emoji = {"US": "🇺🇸", "KR": "🇰🇷", "ALT": "🪙", "FXB": "💱"}.get(v["market"], "")
            position = "🔴 위" if gap >= 0 else "🟢 아래"
            # 고평가 점수/등급 (밸류에이션, 없으면 '-')
            val = v.get("val") or {}
            vscore = val.get("score")
            vgrade = val.get("grade", "-")
            valuation = f"{vgrade} ({vscore})" if vscore is not None else "-"
            # 200일선 복귀 통계 (미국 시총 TOP50만 내장, 달력일 기준)
            rec = _recovery_stats(tk)
            recovery = f"{rec['avg']}/{rec['median']}/{rec['max']}일" if rec else "-"
            hits.append({
                "종목": v["name"],
                "티커": tk,
                "시장": mk_emoji,
                "200일선": position,
                "이격": f"{abs(gap):.1f}%",
                "현재 괴리율": f"{gap:+.1f}%",
                "RSI": _fmt_rsi(r.get("rsi")),
                "고평가": valuation,
                "매칭 구간": f"{int(best_center):+d}%",
                "역사적 승률": f"{wr:.0f}%",
                "표본": f"{samples}건",
                "복귀 평균/중앙/최대": recovery,
                # 정렬/필터용 숨은 숫자값
                "_wr": wr,
                "_gap": gap,
                "_rsi": r.get("rsi") if r.get("rsi") is not None else float("nan"),
                "_samples": samples,
                "_vscore": vscore if vscore is not None else float("nan"),
            })
    prog.empty()

    if not hits:
        extra = " (200일선 복귀 기준은 지금 200일선 아래 종목만 잡혀요)" if mode == "sma" else ""
        st.warning(f"지금 승률 {threshold}% 이상 구간에 있는 종목이 없어요.{extra} "
                   "임계값을 낮추거나 나중에 다시 확인해 보세요.")
        return

    st.success(f"🎯 지금 승률 {threshold}% 이상 구간에 있는 종목: **{len(hits)}개**")

    # --- 정렬 / 필터 위젯 ---
    df_full = pd.DataFrame(hits)
    with st.expander("🔧 정렬 · 필터", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sort_by = st.selectbox("정렬 기준",
                                   ["역사적 승률", "현재 괴리율", "RSI", "표본", "고평가 점수"],
                                   index=0, key="wz_sort")
            sort_desc = st.checkbox("내림차순", value=True, key="wz_sort_desc")
        with fc2:
            gmin, gmax = float(df_full["_gap"].min()), float(df_full["_gap"].max())
            if gmin < gmax:
                gap_range = st.slider("현재 괴리율(%) 범위", gmin, gmax, (gmin, gmax), key="wz_gap")
            else:
                gap_range = (gmin, gmax)
            min_samples = st.number_input("최소 표본 수", min_value=0,
                                          value=0, step=10, key="wz_minsamp")
        with fc3:
            # RSI 범위 (결측 제외 옵션 고려)
            rsi_valid = df_full["_rsi"].dropna()
            if len(rsi_valid) > 0:
                rlo, rhi = int(rsi_valid.min()), int(rsi_valid.max())
                rsi_range = st.slider("RSI 범위", 0, 100, (rlo, rhi), key="wz_rsi")
            else:
                rsi_range = (0, 100)
            only_undervalued = st.checkbox("고평가 제외 (저평가·적정만)",
                                           value=False, key="wz_undervalued")

    # 필터 적용
    f = df_full.copy()
    f = f[(f["_gap"] >= gap_range[0]) & (f["_gap"] <= gap_range[1])]
    f = f[f["_samples"] >= min_samples]
    # 범위 필터는 해당 지표가 있는 종목만 통과시킨다.
    f = f[f["_rsi"].notna() & (f["_rsi"] >= rsi_range[0]) & (f["_rsi"] <= rsi_range[1])]
    if only_undervalued:
        f = f[f["_vscore"].notna() & (f["_vscore"] < 70)]

    # 정렬
    sort_col = {"역사적 승률": "_wr", "현재 괴리율": "_gap", "RSI": "_rsi",
                "표본": "_samples", "고평가 점수": "_vscore"}[sort_by]
    f = f.sort_values(sort_col, ascending=not sort_desc, na_position="last")

    st.caption(f"필터 결과: **{len(f)}개** (전체 {len(df_full)}개 중)")
    # 표시용 컬럼만 (숨은 숫자값 제거)
    show_cols = [c for c in f.columns if not c.startswith("_")]
    st.dataframe(f[show_cols], use_container_width=True, hide_index=True)

    st.caption("· '복귀 평균/중앙/최대'는 200일선 이탈 후 다시 위로 복귀까지 걸린 **달력일**이에요 "
               "(미국 시총 TOP50만 내장, 그 외 종목·코인·환율은 '-'). "
               "· '매칭 구간'은 현재 괴리율을 실제로 포함하는 사전계산 구간이에요. 포함 구간이 없으면 결과에서 제외합니다. "
               f"· 사전계산은 **폭 {meta.get('band', 10):.0f}% / 완충 {meta.get('step', 5):.0f}% 고정 그리드**이고 "
               "표본 15건 이상 구간만 저장해요. '위치별 승률 스크리너'에서 완충을 다르게 두면 같은 종목도 "
               "다른 구간이 잡혀 성공률이 달라 보일 수 있습니다(데이터가 다른 게 아니라 구간이 다른 것). "
               f"· 성공률은 목표 선도달 또는 만기 양수 기준의 사전 백테스트 내장값입니다 (생성일 {generated_date}). "
               "· '고평가'는 PER·PBR·PSR을 같은 시장 내에서 상대 순위로 종합한 점수(0~100, 높을수록 고평가)예요. "
               f"밸류에이션 기준일은 {valuation_date}이며, 미상인 경우 최신성은 보장되지 않습니다. "
               "결측 종목·지수/코인/환율은 '-'로 표시됩니다. "
               "· 표본이 적은 구간은 신뢰도가 낮을 수 있어요. 과거 성과가 미래를 보장하지 않습니다.")


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
                                 "괴리율": "-", "RSI": "-", "상태": "데이터 없음"})
                    continue
                stt = "🟢 200일선 위" if r["above"] else "🔴 200일선 아래"
                if r["signal"] != "-":
                    stt = r["signal"]
                rows.append({
                    "섹터": name, "ETF": tk,
                    "현재가": f"{r['close']:,.2f}",
                    "괴리율": f"{r['gap']:+.1f}%",
                    "RSI": _fmt_rsi(r.get("rsi")), "상태": stt,
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

    st.caption("데이터: Yahoo Finance 섹터 ETF, 1999~2026로 표기된 앱 내 참고값. "
               "이 저장소에는 산식 생성기·정확한 기준일이 없어 독립 재현 검증하지 못했습니다. "
               "XLC(2018)·XLRE(2015)는 표본이 짧음. 순환매는 확률이지 확정이 아니며, "
               "과거 패턴이 미래를 보장하지 않습니다.")


def render_recovery():
    st.subheader("🔄 200일선 복귀 기간")
    st.caption("미국 시총 TOP50이 200일선 아래로 내려간 뒤 얼마 만에 복귀했는지 전수 분석 "
               "(총 7,900 사이클 · 일수는 달력일 기준).")

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
        "행동 가이드": _recovery_action(avg),
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

    st.caption("데이터: Yahoo Finance 전체 기간(종목별 상장~2026.08) · 미국 시총 TOP50 · "
               "총 7,900 이탈→복귀 사이클. 종가가 200일선 아래로 내려간 날부터 다시 위로 올라온 날까지를 "
               "1 사이클로 세고, **아직 복귀하지 않은 진행 중 구간은 제외**했습니다. "
               "**모든 일수는 거래일이 아니라 달력일 기준**이에요(예: 금요일 이탈 → 월요일 복귀 = 3일).  \n"
               "✅ 이 값들은 `recovery_cycles.py`로 실제 가격을 다시 계산해 재현 검증했습니다 "
               "(분포 각 구간 편차 6회 이내, 종목별 최대체류 49/50·중앙값 50/50이 ±1일 이내 일치. "
               "재현 시점이 더 최근이라 사이클 수는 +35회 많습니다). "
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

    with st.spinner("지수·레버리지 ETF 스캔 중..."):
        st.markdown("#### 📊 주요 지수 · 레버리지 ETF 200일선")
        st.caption("주요 지수와 레버리지 ETF가 200일선을 뚫었는지(위/아래·돌파·이탈) 한눈에.")
        st.dataframe(_ds_table(list(_DS_LEV_INDICES.items())), use_container_width=True, hide_index=True)

    with st.spinner("환율·국채 스캔 중..."):
        st.markdown("#### 💱 환율 · 국채 200일선")
        st.caption("달러/원·달러/엔 환율과 미국채(가격·금리)의 200일선 대비 괴리율. "
                   "환율 '위'=원화·엔화 약세, 국채 ETF '아래'=채권가격 하락(금리 상승).")
        st.dataframe(_ds_table(list(_DS_FX_BOND.items())), use_container_width=True, hide_index=True)

    with st.spinner("환율·국채 승률 스캔 중..."):
        st.markdown("---")
        render_fxb_winzone()
        st.markdown("---")

    with st.spinner("미국 대형주 스캔 중..."):
        st.markdown("#### 🇺🇸 미국 주요주")
        st.dataframe(_ds_table([(t, t) for t in _DS_US_TIER1]), use_container_width=True, hide_index=True)

    with st.spinner("국장 대표주 스캔 중..."):
        st.markdown("#### 🇰🇷 국장 대표주")
        st.dataframe(_ds_table(list(_DS_KR.items())), use_container_width=True, hide_index=True)

    with st.spinner("국장 200일선 매수 전략 스캔 중..."):
        st.markdown("---")
        render_kr_winzone()

    with st.spinner("미국 대형주 200일선 매수 전략 스캔 중..."):
        st.markdown("---")
        render_us_winzone()
        st.markdown("---")

    with st.spinner("알트코인 스캔 중..."):
        st.markdown("#### 🪙 알트코인")
        st.dataframe(_ds_table([(t, t.replace("-USD", "")) for t in _DS_ALT]),
                     use_container_width=True, hide_index=True)

    with st.spinner("알트코인 승률 스캔 중..."):
        st.markdown("---")
        render_alt_winzone()

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
    rsi = _rsi_wilder(d["Close"])
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
        "rsi": rsi,
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
            # 우선순위대로 보며, 첫 보유 가능 종목보다 앞선 상태가 미확인일 때만 보류한다.
            target = None
            blocking_asset = None
            for a in _ROTATION_ASSETS:
                s = statuses.get(a["ticker"])
                if s is None:
                    blocking_asset = a
                    break
                if s["holdable"]:
                    target = a
                    break

            if blocking_asset is not None:
                st.warning(f"⚠️ **추천 보류** — 상위 우선순위 **{blocking_asset['name']}** 데이터를 "
                           "확인하지 못했어요. 상태를 모른 채 하위 종목을 추천하지 않습니다.")
            elif target is None:
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
                    rows.append({"우선순위": a["prio"], "종목": a["name"], "기준일": "-",
                                 "현재가": "-", "괴리율": "-", "RSI": "-", "상태": "데이터 없음"})
                    continue
                if target and a["ticker"] == target["ticker"]:
                    stt = "🟢 보유 (현재 타겟)"
                elif s["holdable"]:
                    stt = "🟡 보유 가능 (하위 우선순위)"
                else:
                    stt = "🔴 200일선 아래 (제외)"
                rows.append({
                    "우선순위": a["prio"], "종목": a["name"], "기준일": s["date"],
                    "현재가": f"{s['price']:,.2f}",
                    "괴리율": f"{s['gap']:+.1f}%",
                    "RSI": _fmt_rsi(s.get("rsi")), "상태": stt,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("종목별 기준일 표시 · BTC는 3% 완충 적용 (200일선 -3%까지 보유 유지)")

    # 백테스트 요약 (참고용 내장값)
    with st.expander("📊 백테스트 요약 (2015.07~2026.08, 참고용)", expanded=False):
        st.dataframe(pd.DataFrame(_ROTATION_BACKTEST), use_container_width=True, hide_index=True)
        st.markdown("""
- **갈아타기**가 표 기준 최상위 (배수 15.2배, MAR 2.65). 하위 종목 보유 중 상위가 돌파하면 즉시 전환.
- 단독 전략 중 **BTC(3%완충)**의 MAR 2.39가 **TQQQ 2.23**보다 높습니다. **SOXL**은 단독 절대수익이 가장 높지만 MDD -56%입니다.
- **% 손절 금지**: 레버리지 ETF에 -3% 손절 넣으면 수익이 1/4로 축소. 200일선 이탈만이 매도 신호.
- **종가 기준만**: 장중 이탈로 판단하면 수익 1/16로 축소.
- 200일선 아래로 내려가도 **55%가 1주 이내, 80%가 1개월 이내** 복귀.

<span style='color:gray'>※ 앱 내 참고용 내장값이며, 이 저장소에는 산식 생성기·기준일 메타데이터가 없어 독립 재현 검증하지 못했습니다. Yahoo Finance 기반, 수수료/세금/환율 미반영. 과거 성과가 미래를 보장하지 않습니다.</span>
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
    st.caption("내 평균 단가를 넣으면 각 익절 단계의 목표가를 계산합니다. "
               "실제 매도 이력을 저장하지 않으므로 '도달'은 현재 수익률 기준이며, 이미 실행한 익절 여부는 직접 확인해야 해요.")
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
    sub = st.tabs(["📋 데일리 스캐너", "🎯 승률 포착기", "🌡️ 시장 붕괴 경고"])
    with sub[0]:
        render_daily_screener()
    with sub[1]:
        render_winzone_catcher()
    with sub[2]:
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

**전략 성공률 정의 (목표/손절 방식)**: 각 위치에서 매수한 뒤,
**목표수익에 먼저 닿으면 성공**, **손절에 먼저 닿으면 실패**로 집계해요.
최대 보유기간까지 둘 다 안 닿으면 만기 수익이 플러스일 때도 성공입니다.
완전한 최대 보유기간이 남아 있는 과거 진입만 분석합니다.
- 전략 성공률 = 목표 선도달 또는 만기 양수인 비율
- 평균 수익 = 목표·손절·만기 청산 시 평균 수익률
- 평균 최대도달 = 실제 청산일까지 진입별 최대 수익률의 평균
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
        step = st.selectbox("중심 간격 (완충)", [1, 2, 5, 10], index=1,
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
  그 시점 수익률이 플러스면 성공으로 판정해요. 미래 데이터가 이 기간만큼 온전히 남은 과거 진입만 집계합니다.
</div>
""", unsafe_allow_html=True)

    run = st.button("🔍 전수조사", type="primary", use_container_width=True)

    # --- 한글 기업명 → 종목코드 자동 변환 ---
    # 숫자 6자리 국내 코드는 시장에 맞는 접미사(.KS/.KQ)로 교정해 둔다.
    resolved_ticker = _normalize_kr_code(ticker) if ticker else ""
    resolved_name = None
    proceed = run

    # 버튼을 눌렀으면 이번 조회 티커를 기억(session_state).
    # 이후 박스권 기간 등 다른 위젯을 바꿔 rerun돼도 마지막 조회 결과가 유지됨.
    if run:
        st.session_state["_wz_run"] = True

    should_search_kr = run and ticker and (_has_korean(ticker) or not _looks_like_ticker(ticker))

    if should_search_kr:
        with st.spinner("한국 종목명 검색 중..."):
            kind, payload, *rest = (*resolve_korean_name(ticker), None)

        if kind == "code":
            code, name = payload, rest[0]
            # 코스피/코스닥에 맞는 접미사를 붙인다 (FDR은 접미사를 제거해서 사용)
            resolved_ticker = _kr_ticker(code)
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
                code, name, market = cands[idx]
                resolved_ticker = _kr_ticker(code, market)
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

    # 조회가 확정되면 최종 티커/이름을 session_state에 저장.
    if proceed and resolved_ticker:
        st.session_state["_wz_ticker"] = resolved_ticker
        st.session_state["_wz_name"] = resolved_name
    # 이번에 확정됐거나(proceed), 이전에 조회한 게 있으면(session_state) 결과 표시.
    # 단, 한글 후보 선택 대기 중(should_search_kr True + proceed False)이면 새 조회 시도이므로 제외.
    use_ticker = None
    use_name = None
    if proceed and resolved_ticker:
        use_ticker, use_name = resolved_ticker, resolved_name
    elif not run and not (should_search_kr and not proceed) and st.session_state.get("_wz_ticker"):
        use_ticker = st.session_state["_wz_ticker"]
        use_name = st.session_state.get("_wz_name")

    if use_ticker:
        ticker = use_ticker  # 이후 로직은 확정된 티커 사용
        resolved_name = use_name
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
            cur_rsi = _rsi_wilder(raw["Close"])
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
            c4.metric("200일선 대비", gap_display, help=f"RSI(14): {_fmt_rsi(cur_rsi)}")
            st.caption(f"📉 RSI(14): **{_fmt_rsi(cur_rsi)}** (70↑ 과매수 · 30↓ 과매도)")

            # --- 반전 조짐 (예측 아님, 현재 가격 데이터의 관찰) ---
            rev_label, rev_reasons = _reversal_signal(raw["Close"], cur_gap)
            rev_detail = (" · 근거: " + ", ".join(rev_reasons)) if rev_reasons else ""
            st.caption(f"🔁 **반전 조짐**: {rev_label}{rev_detail}  \n"
                       "<span style='color:gray'>※ 미래 예측이 아니라 지금 RSI·괴리율에서 나타나는 신호예요. "
                       "조짐이 있어도 추세가 더 갈 수 있습니다.</span>",
                       unsafe_allow_html=True)

            # --- 현재 추세 (50일선 vs 200일선) ---
            cur_sma50 = float(df["SMA50"].iloc[-1]) if "SMA50" in df.columns else None
            if cur_sma50 is not None:
                if cur_sma50 > cur_sma:
                    trend_txt = "🔼 **상승추세** (50일선 > 200일선)"
                    trend_box = st.success
                else:
                    trend_txt = "🔽 **하락추세** (50일선 < 200일선)"
                    trend_box = st.error
                gap5020 = (cur_sma50 / cur_sma - 1) * 100
                trend_box(f"{trend_txt} · 50일선 {cur_sma50:,.2f} "
                          f"(200일선 대비 {gap5020:+.1f}%)  \n"
                          f"→ 위 표에서 이 종목의 **{'🔼상승추세' if cur_sma50 > cur_sma else '🔽하락추세'} 열**을 보세요.")

            st.markdown(
                f"📅 **데이터 기간**: {raw_start} ~ {raw_end} "
                f"(약 {raw_years:.1f}년, {len(raw):,} 거래일)  \n"
                f"🔎 **분석 구간**: {analysis_start} ~ {last_date} "
                f"({total_days:,} 거래일) "
                f"<span style='color:gray'>· 앞 199거래일은 첫 200일선 계산에 사용되어 분석에서 제외</span>",
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

            # --- 밸류에이션 · 재무건전성 (실시간) ---
            with st.spinner("재무 지표 조회 중..."):
                fund = get_fundamentals(fav_ticker)
            val_summary, val_details = _fmt_valuation(fund)
            health_summary, health_details = _fmt_health(fund)

            if val_details or health_details:
                st.markdown("#### 💼 밸류에이션 · 재무건전성")
                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    st.markdown(f"**고평가 여부: {val_summary}**")
                    if val_details:
                        st.dataframe(
                            pd.DataFrame(val_details, columns=["지표", "값", "판단"]),
                            use_container_width=True, hide_index=True)
                    else:
                        st.caption("밸류에이션 지표 없음 (코인/지수 등)")
                with fcol2:
                    st.markdown(f"**재무건전성: {health_summary}**")
                    if health_details:
                        st.dataframe(
                            pd.DataFrame(health_details, columns=["지표", "값", "판단"]),
                            use_container_width=True, hide_index=True)
                    else:
                        st.caption("재무 지표 없음 (코인/지수/일부 종목)")
                st.caption("· PER≥30·PBR≥5·PSR≥10 = 높음(고평가 신호). "
                           "· ROE≥15%·부채비율<100%·영업이익률≥20% = 우수. "
                           "yfinance 실시간 값이며 결측일 수 있어요. 투자 권유가 아닙니다.")
            else:
                st.caption(f"💼 `{fav_ticker}`의 밸류에이션·재무 지표를 받지 못했어요. "
                           "코인·지수·환율은 원래 해당 지표가 없고, 개별 종목이라면 "
                           "yfinance가 그 종목 재무를 제공하지 않는 경우예요.")

            # --- 박스권 자동 탐지 ---
            st.markdown("#### 📦 박스권 자동 탐지")
            box_days = st.selectbox("탐지 기간", [40, 60, 90, 120], index=1, key="box_days",
                                    format_func=lambda x: f"최근 {x}일")
            box = detect_box(df, lookback=box_days)
            if box is None:
                st.caption("데이터가 부족해 박스권을 탐지할 수 없어요.")
            else:
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("박스 상단", f"{box['top']:,.2f}")
                b2.metric("박스 하단", f"{box['bottom']:,.2f}")
                b3.metric("박스 폭", f"{box['width_pct']:.1f}%")
                b4.metric("박스 기준 위치", f"{box['pos_pct']:.0f}%")
                if box["is_box"]:
                    st.info(f"📦 **박스권입니다** (최근 {box_days}일 횡보) · {box['status']}  \n"
                            f"상단 **{box['top']:,.2f}** / 하단 **{box['bottom']:,.2f}** "
                            f"(폭 {box['width_pct']:.1f}%). 현재가는 박스 내 {box['pos_pct']:.0f}% 지점.")
                else:
                    st.warning(f"↗️ **박스권이 아니에요** (추세 진행 중, 최근 {box_days}일 방향성 {box['trend_pct']:.1f}%). "
                               f"{box['status']} · 최근 고점 {box['top']:,.2f} / 저점 {box['bottom']:,.2f}")
                st.caption("· 박스권 = 좁은 범위 횡보 (폭 25%↓ + 방향성 8%↓). "
                           "· 상단 돌파는 매수 신호, 하단 이탈은 매도 신호로 해석돼요. "
                           "· 과거 데이터 기반이며 투자 권유가 아닙니다.")

            st.markdown(f"🎯 목표 **+{target_pct:.0f}%** / 🛑 손절 **-{stop_pct:.0f}%** | "
                        f"최대보유: **{max_hold_choice}** | "
                        f"구간 폭: **{band_width}%** / 완충: **{step}%**")

            # --- 구간별 전수조사 ---
            with st.spinner("전수조사 계산 중..."):
                result = zone_analysis(df, band_width, step, target_pct, stop_pct, max_hold)

            if result.empty:
                st.warning("분석할 데이터가 부족합니다.")
            else:
                # 현재 괴리율을 실제로 포함하는 표본 구간 중 가장 가까운 중심만 선택한다.
                cur_center_key = _match_zone_center(result["center"].tolist(), cur_gap, band_width)
                if cur_center_key is None:
                    cur_zone_idx = None
                    cur_center = None
                else:
                    cur_center = float(cur_center_key)
                    cur_zone_idx = int(result.index[result["center"] == cur_center_key][0])

                st.markdown(f"### 📊 [핵심] 200일선 대비 위치별 전략 성공률")
                if cur_zone_idx is None:
                    st.warning(
                        f"⚠️ 현재 괴리율(**{cur_gap:+.1f}%**)을 포함하면서 과거 표본도 있는 구간이 없어요. "
                        "가장 가까운 다른 구간으로 강제 대체하지 않으며, 현재 표시와 결론을 생략합니다.")
                st.markdown(f"폭 {band_width}%, 완충 {step}%, 목표 +{target_pct:.0f}% / 손절 -{stop_pct:.0f}%, "
                            f"미도달 시 만기 수익 부호 판정, 최대보유 {max_hold_choice} 기준 전수조사 결과:  \n"
                            f"<span style='color:gray'>· '해당 가격'은 현재 200일선({cur_sma:,.2f}) 기준 그 위치 가격이에요.  \n"
                            f"· <b>전략 성공률</b> = +{target_pct:.0f}% 목표 선도달, 또는 목표·손절 미도달 시 만기 수익이 플러스인 비율.  \n"
                            f"· <b>🔼상승추세 / 🔽하락추세</b> = 같은 성공률을 <b>추세별로 분리</b> (50일선>200일선=상승추세). "
                            f"괄호 안은 각 추세의 표본 수예요.  \n"
                            f"· <b>200선복귀 성공률</b> = 진입 괴리율 <b>-3% 이하</b>에서 매수 후 {max_hold_choice} 내 "
                            f"종가가 200일선 위로 복귀한 비율.  \n"
                            f"· <b>이탈매도 성공률</b> = 진입 당시 -5% 이탈선 위인 표본만 대상으로, "
                            f"종가가 200일선 -5% 아래로 이탈할 때(미이탈 시 만기) 수익이 플러스인 비율. "
                            f"모든 전략은 완전한 최대 보유기간이 남은 진입만 집계합니다.</span>",
                            unsafe_allow_html=True)

                # 표시용 DataFrame 구성 (네이티브 st.dataframe → 스크롤 안정)
                def _pct_or_dash(v, n=None):
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    if n is not None and int(n or 0) == 0:
                        return None
                    return float(v)

                disp_rows = []
                for i, row in result.iterrows():
                    is_current = (i == cur_zone_idx)
                    center_label = f"{row['center']:+.0f}%"
                    if abs(row["center"]) < 0.01:
                        center_label = "0% (200일선)"
                    if is_current:
                        center_label += " ◀ 현재"

                    up_v = _pct_or_dash(row.get("up_win_rate"), row.get("up_trades"))
                    up_n = int(row.get("up_trades") or 0)
                    down_v = _pct_or_dash(row.get("down_win_rate"), row.get("down_trades"))
                    down_n = int(row.get("down_trades") or 0)

                    disp_rows.append({
                        "중심위치": center_label,
                        "해당가격": cur_sma * (1 + row["center"] / 100),
                        "구간": row["zone_label"],
                        "거래수": int(row["trades"]),
                        "승률(목표/손절)": float(row["win_rate"]),
                        "🔼상승추세": up_v,
                        "🔼표본": up_n if up_n else None,
                        "🔽하락추세": down_v,
                        "🔽표본": down_n if down_n else None,
                        "승률(200선복귀)": _pct_or_dash(row.get("sma_win_rate")),
                        "200선표본": int(row.get("sma_trades") or 0) or None,
                        "승률(이탈매도)": _pct_or_dash(row.get("breach_win_rate")),
                        "이탈표본": int(row.get("breach_trades") or 0) or None,
                        "평균수익": float(row["avg_return"]),
                        "최대수익": float(row["max_return"]),
                        "평균보유": int(row["avg_holding_days"]),
                    })

                disp_df = pd.DataFrame(disp_rows)

                # 정적 HTML 테이블 (내부 스크롤 컨테이너 없음 → 모바일 터치 스크롤 충돌 방지)
                # 셀마다 인라인 스타일 대신 CSS 클래스만 붙여 HTML을 가볍게 유지
                def _wr_cls(v):
                    if v is None or pd.isna(v):
                        return "wr-na"
                    return "wr-hi" if v >= 60 else ("wr-mid" if v >= 45 else "wr-lo")

                def _wr_txt(v):
                    return "-" if (v is None or pd.isna(v)) else f"{v:.0f}%"

                _rows_html = []
                for _, r in disp_df.iterrows():
                    is_cur = "◀ 현재" in str(r["중심위치"])
                    tr_cls = ' class="cur"' if is_cur else ""
                    up_txt = _wr_txt(r["🔼상승추세"])
                    up_n = "" if pd.isna(r["🔼표본"]) else f'<span class="n">({int(r["🔼표본"])})</span>'
                    dn_txt = _wr_txt(r["🔽하락추세"])
                    dn_n = "" if pd.isna(r["🔽표본"]) else f'<span class="n">({int(r["🔽표본"])})</span>'
                    sma_n = "" if pd.isna(r["200선표본"]) else f'<span class="n">({int(r["200선표본"])})</span>'
                    br_n = "" if pd.isna(r["이탈표본"]) else f'<span class="n">({int(r["이탈표본"])})</span>'
                    ar = r["평균수익"]
                    ar_cls = "ret-up" if ar > 0 else ("ret-dn" if ar < 0 else "ret-0")
                    _rows_html.append(
                        f'<tr{tr_cls}>'
                        f'<td class="lft">{r["중심위치"]}</td>'
                        f'<td class="lft price">{r["해당가격"]:,.2f}</td>'
                        f'<td class="lft dim">{r["구간"]}</td>'
                        f'<td>{int(r["거래수"]):,}</td>'
                        f'<td class="{_wr_cls(r["승률(목표/손절)"])}">{_wr_txt(r["승률(목표/손절)"])}</td>'
                        f'<td class="{_wr_cls(r["🔼상승추세"])}">{up_txt}{up_n}</td>'
                        f'<td class="{_wr_cls(r["🔽하락추세"])}">{dn_txt}{dn_n}</td>'
                        f'<td class="{_wr_cls(r["승률(200선복귀)"])}">{_wr_txt(r["승률(200선복귀)"])}{sma_n}</td>'
                        f'<td class="{_wr_cls(r["승률(이탈매도)"])}">{_wr_txt(r["승률(이탈매도)"])}{br_n}</td>'
                        f'<td class="{ar_cls}">{ar:+.1f}%</td>'
                        f'<td class="ret-max">{r["최대수익"]:+.1f}%</td>'
                        f'<td class="dim">{int(r["평균보유"])}일</td>'
                        f'</tr>'
                    )

                _head = (
                    '<tr>'
                    '<th>중심위치</th><th>해당가격</th><th>구간</th><th>거래수</th>'
                    '<th>성공률<br><span class="sub">목표/손절</span></th>'
                    '<th>성공률<br><span class="sub">🔼상승</span></th>'
                    '<th>성공률<br><span class="sub">🔽하락</span></th>'
                    '<th>성공률<br><span class="sub">200선복귀(n)</span></th>'
                    '<th>성공률<br><span class="sub">이탈매도(n)</span></th>'
                    '<th>평균수익</th><th>평균<br><span class="sub">최대도달</span></th><th>평균보유</th>'
                    '</tr>'
                )
                st.markdown(
                    '<div class="posbox"><table class="postbl">'
                    + _head + "".join(_rows_html)
                    + "</table></div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "※ **평균수익·평균 최대도달·평균보유는 모두 목표/손절/만기 전략 기준**입니다. "
                    "평균 최대도달은 각 진입에서 **진입 시점 0%를 포함해 실제 청산일까지** 기록한 최대 수익률의 평균이에요.  \n"
                    "※ 200선복귀·이탈매도 셀의 괄호는 각 전략의 별도 표본 수입니다. "
                    "**200선복귀**는 진입 괴리율 -3% 이하만, **이탈매도**는 진입 당시 -5% 이탈선 위만 집계합니다. "
                    "따라서 성공률과 표본이 기본 전략과 다르며 평균수익·평균보유를 서로 직접 연결하면 안 됩니다.  \n"
                    "※ 표본은 **하루 단위 진입**이라 같은 국면의 연속된 날들이 여러 건으로 잡혀요. "
                    "예컨대 '8건'이 실제로는 며칠짜리 하락 한두 번일 수 있으니, 표본이 적은 구간의 "
                    "0%·100%는 확률이 아니라 **그 몇 번의 결과**로 읽으세요."
                )

                # --- 현재 위치 결론 ---
                if cur_zone_idx is not None:
                    cur_row = result.iloc[cur_zone_idx]
                    wr = cur_row["win_rate"]
                    rule = f"목표 +{target_pct:.0f}% / 손절 -{stop_pct:.0f}%, 표본 {cur_row['trades']}건"
                    st.markdown("---")
                    if wr >= 60:
                        st.success(
                            f"🟢 현재 위치({cur_gap:+.1f}%)의 **전략 성공률이 높았어요**. "
                            f"성공률 **{wr:.0f}%**, 평균 청산수익 **{cur_row['avg_return']:+.1f}%**, "
                            f"평균 보유 **{cur_row['avg_holding_days']}일** ({rule})"
                        )
                    elif wr >= 45:
                        st.warning(
                            f"🟡 현재 위치({cur_gap:+.1f}%)의 **전략 성공률은 중간 수준**이에요. "
                            f"성공률 **{wr:.0f}%**, 평균 청산수익 **{cur_row['avg_return']:+.1f}%**, "
                            f"평균 보유 **{cur_row['avg_holding_days']}일** ({rule})"
                        )
                    else:
                        st.error(
                            f"🔴 현재 위치({cur_gap:+.1f}%)의 **전략 성공률이 낮았어요**. "
                            f"성공률 **{wr:.0f}%**, 평균 청산수익 **{cur_row['avg_return']:+.1f}%**, "
                            f"평균 보유 **{cur_row['avg_holding_days']}일** ({rule})"
                        )

                # --- 가격 차트 (항상 표시) ---
                st.markdown("### 📉 가격 vs 200일선")
                chart_df = df[["Close", "SMA200"]].rename(columns={"Close": "종가", "SMA200": "200일선"})
                st.line_chart(chart_df)

st.divider()
st.caption("⚠️ 과거 데이터 기반 통계이며 투자 권유가 아닙니다. 과거 성과가 미래를 보장하지 않습니다.")
