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


def zone_winrates(close: np.ndarray):
    """구간별 (승률, 표본수) dict 반환."""
    n = len(close)
    sma = pd.Series(close).rolling(200).mean().values
    valid_sma = ~np.isnan(sma)
    gap = np.full(n, np.nan)
    gap[valid_sma] = (close[valid_sma] / sma[valid_sma] - 1) * 100

    # pos별 결과 1회 계산
    exit_win = np.zeros(n, dtype=bool)
    has = np.zeros(n, dtype=bool)
    for pos in range(n - 1):
        if np.isnan(gap[pos]):
            continue
        entry = close[pos]
        end = min(pos + MAX_HOLD, n - 1)
        path = close[pos + 1:end + 1]
        if path.size == 0:
            continue
        cum = (path / entry - 1.0) * 100.0
        hit_t = np.argmax(cum >= TARGET_PCT) if (cum >= TARGET_PCT).any() else -1
        hit_s = np.argmax(cum <= -STOP_PCT) if (cum <= -STOP_PCT).any() else -1
        has[pos] = True
        if hit_t != -1 and (hit_s == -1 or hit_t <= hit_s):
            exit_win[pos] = True
        elif hit_s != -1:
            exit_win[pos] = False
        else:
            exit_win[pos] = cum[-1] > 0

    half = BAND_WIDTH / 2
    n_neg = int(np.floor((0 - ZONE_MIN) / STEP))
    n_pos = int(np.floor((ZONE_MAX - 0) / STEP))
    centers = [round(-k * STEP, 6) for k in range(n_neg, 0, -1)] + \
              [round(k * STEP, 6) for k in range(0, n_pos + 1)]

    out = {}
    for center in centers:
        lo, hi = center - half, center + half
        sel = has & (gap >= lo) & (gap < hi)
        t = int(sel.sum())
        if t >= MIN_SAMPLES:
            wr = float(exit_win[sel].mean() * 100)
            out[int(center)] = [round(wr, 1), t]
    return out


def get_us_top100():
    import FinanceDataReader as fdr
    sp = fdr.StockListing("S&P500")
    tickers = sp["Symbol"].tolist()
    names = dict(zip(sp["Symbol"], sp["Name"]))
    # 시총으로 정렬하려면 yfinance 조회 필요 -> 시간 절약 위해 marketCap 배치 조회
    caps = {}
    print(f"  미장 시총 조회 중 ({len(tickers)}개)...")
    for i, tk in enumerate(tickers):
        try:
            info = yf.Ticker(tk).fast_info
            caps[tk] = info.get("market_cap") or 0
        except Exception:
            caps[tk] = 0
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(tickers)}")
    top = sorted(tickers, key=lambda t: caps.get(t, 0), reverse=True)[:100]
    return [(tk, names.get(tk, tk)) for tk in top]


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


def main():
    print("종목 리스트 확보 중...")
    us = get_us_top100()
    kr = get_kr_top100()
    print(f"  미장 {len(us)}개, 국장 {len(kr)}개")

    result = {}  # ticker -> {"name":..., "market":"US"/"KR", "zones": {...}}
    all_items = [(tk, nm, "US") for tk, nm in us] + [(tk, nm, "KR") for tk, nm in kr]

    for i, (tk, nm, mk) in enumerate(all_items):
        close = load_close(tk)
        if close is None:
            print(f"  [{i+1}/{len(all_items)}] {tk} {nm}: 데이터 부족, 스킵")
            continue
        zones = zone_winrates(close.values.astype(float))
        if zones:
            result[tk] = {"name": nm, "market": mk, "zones": zones}
        print(f"  [{i+1}/{len(all_items)}] {tk} {nm}: 구간 {len(zones)}개")
        time.sleep(0.05)

    # winzone_data.py 로 저장
    with open("winzone_data.py", "w", encoding="utf-8") as f:
        f.write('"""승률 포착기 사전 계산 데이터 (precompute_winzone.py 로 생성).\n')
        f.write(f'기준: 목표+{TARGET_PCT:.0f}% / 손절-{STOP_PCT:.0f}% / 최대보유 {MAX_HOLD}거래일\n')
        f.write('각 종목 zones: {중심위치%: [승률%, 표본수]}\n"""\n')
        f.write(f"WINZONE_META = {{'target': {TARGET_PCT}, 'stop': {STOP_PCT}, 'max_hold': {MAX_HOLD}, "
                f"'band': {BAND_WIDTH}, 'step': {STEP}}}\n\n")
        f.write("WINZONE_DATA = ")
        f.write(json.dumps(result, ensure_ascii=False, indent=0))
        f.write("\n")
    print(f"\n완료: {len(result)}개 종목 -> winzone_data.py")


if __name__ == "__main__":
    main()
