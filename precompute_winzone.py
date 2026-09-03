"""
승률 포착기용 사전 백테스트 (로컬에서 1회 실행 → winzone_data.py 생성)

미장 TOP100(S&P500 시총순) + 국장 TOP100(KRX 시총순) 각 종목에 대해:
  - 200일선 대비 위치를 5% 구간(중심 간격 5%, 폭 10% 슬라이딩)으로 나누고
  - 목표 +10% / 손절 -5% / 최대보유 3개월(63거래일) 기준 승률 계산
결과를 winzone_data.py 에 파이썬 상수로 저장 (앱에 내장).

승률 데이터: {center: [win_rate, trades], ...}  (표본 15건 이상만 저장)
"""
import sys
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf

TARGET_PCT = 10.0
STOP_PCT = 5.0
MAX_HOLD = 63
BAND_WIDTH = 10.0
STEP = 5.0
ZONE_MIN, ZONE_MAX = -50, 50
MIN_SAMPLES = 15  # 이 이상 표본이 있는 구간만 저장


def load_close(ticker):
    try:
        raw = yf.download(ticker, start="2000-01-01", auto_adjust=True, progress=False)
        if raw is None or len(raw) == 0:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if "Close" not in raw.columns:
            return None
        s = raw["Close"].dropna()
        return s if len(s) >= 260 else None
    except Exception:
        return None


def fetch_valuation(ticker):
    """yfinance .info에서 밸류에이션 지표 수집. 결측은 None.
    Returns {per, pbr, psr, div} (div=배당수익률%)."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return {"per": None, "pbr": None, "psr": None, "div": None}

    def g(*keys):
        for k in keys:
            v = info.get(k)
            if v is not None and isinstance(v, (int, float)) and v == v and v > 0:
                return float(v)
        return None

    return {
        "per": g("trailingPE", "forwardPE"),
        "pbr": g("priceToBook"),
        "psr": g("priceToSalesTrailing12Months"),
    }


def zone_winrates(close: np.ndarray):
    """구간별 승률을 두 방식으로 계산해 (zones_target, zones_sma) 반환.

    zones_target: 목표 +TARGET_PCT% 익절 vs -STOP_PCT% 손절 (먼저 닿는 쪽). 전 구간.
    zones_sma   : 200일선 복귀 시 매도. 200일선 아래(-3% 이하) 진입만 의미 있음.
                  MAX_HOLD 내 200일선 위로 복귀하면 승리, 아니면 패배.
    각 dict: {중심위치%: [승률%, 표본수]}
    """
    n = len(close)
    sma = pd.Series(close).rolling(200).mean().values
    valid_sma = ~np.isnan(sma)
    gap = np.full(n, np.nan)
    gap[valid_sma] = (close[valid_sma] / sma[valid_sma] - 1) * 100

    # pos별 결과 1회 계산 (두 방식 동시)
    win_target = np.zeros(n, dtype=bool)
    has_target = np.zeros(n, dtype=bool)
    win_sma = np.zeros(n, dtype=bool)
    has_sma = np.zeros(n, dtype=bool)

    for pos in range(n - 1):
        if np.isnan(gap[pos]):
            continue
        entry = close[pos]
        end = min(pos + MAX_HOLD, n - 1)
        path = close[pos + 1:end + 1]
        if path.size == 0:
            continue
        cum = (path / entry - 1.0) * 100.0

        # 방식 1: 목표/손절
        hit_t = np.argmax(cum >= TARGET_PCT) if (cum >= TARGET_PCT).any() else -1
        hit_s = np.argmax(cum <= -STOP_PCT) if (cum <= -STOP_PCT).any() else -1
        has_target[pos] = True
        if hit_t != -1 and (hit_s == -1 or hit_t <= hit_s):
            win_target[pos] = True
        elif hit_s != -1:
            win_target[pos] = False
        else:
            win_target[pos] = cum[-1] > 0

        # 방식 2: 200일선 복귀 (200일선 아래 진입만)
        if gap[pos] <= -3:
            fut_sma = sma[pos + 1:end + 1]
            fut_close = path
            recovered = fut_close > fut_sma  # 종가가 200일선 위로
            has_sma[pos] = True
            win_sma[pos] = bool(recovered.any())  # 3개월 내 한 번이라도 복귀하면 승리

    half = BAND_WIDTH / 2
    n_neg = int(np.floor((0 - ZONE_MIN) / STEP))
    n_pos = int(np.floor((ZONE_MAX - 0) / STEP))
    centers = [round(-k * STEP, 6) for k in range(n_neg, 0, -1)] + \
              [round(k * STEP, 6) for k in range(0, n_pos + 1)]

    zones_target, zones_sma = {}, {}
    for center in centers:
        lo, hi = center - half, center + half
        in_zone = (gap >= lo) & (gap < hi)

        sel_t = has_target & in_zone
        t = int(sel_t.sum())
        if t >= MIN_SAMPLES:
            zones_target[int(center)] = [round(float(win_target[sel_t].mean() * 100), 1), t]

        sel_s = has_sma & in_zone
        s = int(sel_s.sum())
        if s >= MIN_SAMPLES:
            zones_sma[int(center)] = [round(float(win_sma[sel_s].mean() * 100), 1), s]

    return zones_target, zones_sma


# 미국 시총 대형주 100 (안정성 위해 하드코딩. yfinance fast_info 시총 조회가 불안정해서)
US_TOP100_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "LLY",
    "JPM", "WMT", "V", "MA", "XOM", "ORCL", "UNH", "COST", "HD", "PG",
    "JNJ", "NFLX", "ABBV", "BAC", "CRM", "CVX", "KO", "AMD", "TMUS", "PEP",
    "WFC", "LIN", "CSCO", "ACN", "MCD", "ADBE", "IBM", "MRK", "ABT", "GE",
    "PM", "TXN", "ISRG", "QCOM", "DIS", "CAT", "VZ", "INTU", "GS", "T",
    "BKNG", "AMGN", "SPGI", "RTX", "NOW", "UBER", "PGR", "MS", "NEE", "HON",
    "LOW", "UNP", "AMAT", "BLK", "SCHW", "TJX", "SYK", "C", "BSX", "COP",
    "DHR", "PLD", "VRTX", "ADP", "BA", "MDT", "GILD", "MMC", "ADI", "LRCX",
    "ETN", "CB", "MU", "KLAC", "AMT", "SBUX", "ANET", "PANW", "INTC", "MO",
    "SO", "ELV", "ICE", "KKR", "REGN", "DUK", "PYPL", "APH", "CI", "SHW",
]


def get_us_top100():
    import FinanceDataReader as fdr
    names = {}
    try:
        sp = fdr.StockListing("S&P500")
        names = dict(zip(sp["Symbol"], sp["Name"]))
    except Exception:
        pass
    return [(tk, names.get(tk, tk)) for tk in US_TOP100_TICKERS]


def get_kr_top100():
    import FinanceDataReader as fdr
    df = fdr.StockListing("KRX")
    df = df[df["Marcap"].notna()].sort_values("Marcap", ascending=False)
    # 우선주(코드 끝 5,7,9 등) 제외 대충 필터: 이름에 '우' 끝나는 것 제외
    out = []
    for _, r in df.iterrows():
        name = str(r["Name"])
        if name.endswith("우") or name.endswith("우B"):
            continue
        code = str(r["Code"])
        market = str(r["Market"])
        suffix = ".KQ" if "KOSDAQ" in market else ".KS"
        out.append((f"{code}{suffix}", name))
        if len(out) >= 100:
            break
    return out


def get_alts():
    """주요 알트코인 (앱의 데일리 스캐너 목록과 동일)."""
    return [
        ("BTC-USD", "BTC"), ("ETH-USD", "ETH"), ("SOL-USD", "SOL"),
        ("DOGE-USD", "DOGE"), ("XRP-USD", "XRP"), ("ADA-USD", "ADA"),
        ("AVAX-USD", "AVAX"), ("LINK-USD", "LINK"), ("BNB-USD", "BNB"),
    ]


def get_fx_bond():
    """환율·국채 (앱의 _DS_FX_BOND 목록과 동일)."""
    return [
        ("KRW=X",     "달러/원(USDKRW)"),
        ("JPY=X",     "달러/엔(USDJPY)"),
        ("DX-Y.NYB",  "달러인덱스(DXY)"),
        ("SHY",       "미국채1-3년(SHY)"),
        ("IEF",       "미국채7-10년(IEF)"),
        ("TLT",       "미국채20년+(TLT)"),
        ("^TNX",      "미국채10년금리"),
        ("^TYX",      "미국채30년금리"),
    ]


def get_indices():
    """주요 지수 (강세장 필터용). market='IDX'."""
    return [
        ("^KS11",  "코스피 지수"),
        ("^KQ11",  "코스닥 지수"),
        ("^IXIC",  "나스닥 종합"),
        ("^GSPC",  "S&P500 지수"),
    ]


def _compute_valuation_scores(result):
    """수집된 밸류에이션을 시장(US/KR) 내 상대순위로 고평가 점수(0~100)+등급 계산.
    PER/PBR/PSR 높을수록 고평가(↑), 배당수익률 높을수록 저평가(↓).
    각 지표 퍼센타일 평균 → 점수. 각 종목 entry['val']에 score/grade 추가."""
    import numpy as _np

    for market in ("US", "KR"):
        items = [(tk, v) for tk, v in result.items()
                 if v.get("market") == market and v.get("val")]
        if len(items) < 5:
            continue

        # 지표별 값 배열 (결측 제외하고 퍼센타일 계산용)
        metrics = {"per": [], "pbr": [], "psr": []}
        for _, v in items:
            for m in metrics:
                val = v["val"].get(m)
                if val is not None:
                    metrics[m].append(val)

        def pct_rank(val, arr):
            """arr 내 val의 퍼센타일(0~100). 높을수록 고평가."""
            if val is None or not arr:
                return None
            arr_sorted = sorted(arr)
            below = sum(1 for x in arr_sorted if x < val)
            return below / len(arr_sorted) * 100

        for tk, v in items:
            val = v["val"]
            parts = []
            for m in ("per", "pbr", "psr"):
                p = pct_rank(val.get(m), metrics[m])
                if p is not None:
                    parts.append(p)

            if parts:
                score = round(float(_np.mean(parts)), 0)
                if score >= 70:
                    grade = "🔴 고평가"
                elif score >= 40:
                    grade = "🟡 적정"
                else:
                    grade = "🟢 저평가"
                val["score"] = int(score)
                val["grade"] = grade
            else:
                val["score"] = None
                val["grade"] = "-"


def main():
    print("종목 리스트 확보 중...")
    us = get_us_top100()
    kr = get_kr_top100()
    alt = get_alts()
    fxb = get_fx_bond()
    idx = get_indices()
    print(f"  미장 {len(us)}개, 국장 {len(kr)}개, 알트 {len(alt)}개, "
          f"환율/국채 {len(fxb)}개, 지수 {len(idx)}개")

    result = {}
    all_items = ([(tk, nm, "US") for tk, nm in us]
                 + [(tk, nm, "KR") for tk, nm in kr]
                 + [(tk, nm, "ALT") for tk, nm in alt]
                 + [(tk, nm, "FXB") for tk, nm in fxb]
                 + [(tk, nm, "IDX") for tk, nm in idx])

    for i, (tk, nm, mk) in enumerate(all_items):
        close = load_close(tk)
        if close is None:
            print(f"  [{i+1}/{len(all_items)}] {tk} {nm}: 데이터 부족, 스킵")
            continue
        zones_t, zones_s = zone_winrates(close.values.astype(float))
        if zones_t or zones_s:
            entry = {"name": nm, "market": mk,
                     "zones": zones_t, "zones_sma": zones_s}
            # 주식(US/KR)만 밸류에이션 수집 (지수/환율/국채/코인은 의미 없음)
            if mk in ("US", "KR"):
                entry["val"] = fetch_valuation(tk)
            result[tk] = entry
        print(f"  [{i+1}/{len(all_items)}] {tk} {nm}: "
              f"목표/손절 {len(zones_t)}구간, 200선복귀 {len(zones_s)}구간")
        time.sleep(0.05)

    # --- 밸류에이션 상대순위 → 고평가 점수(0~100) 계산 ---
    _compute_valuation_scores(result)

    # winzone_data.py 로 저장 (호환성 유지)
    with open("winzone_data.py", "w", encoding="utf-8") as f:
        f.write('"""승률 포착기 사전 계산 데이터 (precompute_winzone.py 로 생성).\n')
        f.write(f'기준: 목표+{TARGET_PCT:.0f}% / 손절-{STOP_PCT:.0f}% / 최대보유 {MAX_HOLD}거래일\n')
        f.write('각 종목 zones: {중심위치%: [승률%, 표본수]}\n"""\n')
        f.write(f"WINZONE_META = {{'target': {TARGET_PCT}, 'stop': {STOP_PCT}, 'max_hold': {MAX_HOLD}, "
                f"'band': {BAND_WIDTH}, 'step': {STEP}}}\n\n")
        f.write("WINZONE_DATA = ")
        f.write(json.dumps(result, ensure_ascii=False, indent=0))
        f.write("\n")

    # winzone_data.json 으로도 저장 (앱은 이걸 사용 — 로딩이 가볍고 안정적)
    payload = {
        "meta": {"target": TARGET_PCT, "stop": STOP_PCT, "max_hold": MAX_HOLD,
                 "band": BAND_WIDTH, "step": STEP},
        "data": result,
    }
    with open("winzone_data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))
    print(f"\n완료: {len(result)}개 종목 -> winzone_data.py + winzone_data.json")


if __name__ == "__main__":
    main()
