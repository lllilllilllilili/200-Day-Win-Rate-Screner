"""
'복귀 빠른 눌림목' 스캐너용 사전계산 (로컬 1회 실행 → recovery_data.json 생성)

winzone_data.json 의 미장/국장 대형주(각 TOP100)를 대상으로 세 가지를 계산한다:

1) 200일선 이탈→복귀 사이클 통계 (달력일 기준, 미복귀 구간 제외)
   - 최단/중간(중앙값)/최장, 평균, 1주 이내 복귀율, 2주 초과 체류율
   - recovery_cycles.py 와 동일한 정의를 사용한다.
2) 지수 민감도 (최근 3년 일간수익률 기준 베타·상관계수)
   - 미장 ^GSPC / 코스피 ^KS11 / 코스닥 ^KQ11 대비
   - 베타가 낮을수록 '지수 등락에 덜 휩쓸리는' 종목.
3) 재무건전성 (yfinance .info: ROE / 부채비율 / 영업이익률) + 시가총액
   - 앱의 _fmt_health 와 같은 기준으로 good/3 을 세어 등급을 매긴다.

실행: python precompute_recovery.py
"""
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

SMA_WINDOW = 200
BETA_YEARS = 3
MIN_BETA_DAYS = 250          # 베타 계산 최소 표본
HISTORY_START = "2000-01-01"

# 시장별 기준 지수
INDEX_FOR = {"US": "^GSPC", "KS": "^KS11", "KQ": "^KQ11"}


def _flatten(raw):
    if raw is None or len(raw) == 0:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if "Close" not in raw.columns:
        return None
    s = raw["Close"].dropna().sort_index()
    return s if len(s) else None


def load_close(ticker, start=HISTORY_START):
    """상장 전체 기간 수정종가.

    복귀 사이클은 상장 이후 전체 기간을 봐야 앱 내장값(_RECOVERY_TOP50)과
    기준이 같아지므로 period=max 를 먼저 시도한다.
    """
    for kwargs in ({"period": "max"}, {"start": start}):
        try:
            s = _flatten(yf.download(ticker, auto_adjust=True, progress=False, **kwargs))
        except Exception:
            s = None
        if s is not None and len(s) > SMA_WINDOW:
            return s
    return None


def recovery_stats(close: pd.Series):
    """200일선 이탈→복귀 사이클 통계 (달력일). 미복귀 구간은 제외."""
    sma = close.rolling(SMA_WINDOW).mean()
    valid = sma.notna()
    px, ma = close[valid].to_numpy(float), sma[valid].to_numpy(float)
    dates = close.index[valid]

    durations = []
    start_idx = None
    for i in range(len(px)):
        if start_idx is None:
            if px[i] < ma[i]:
                start_idx = i
        elif px[i] > ma[i]:
            durations.append((dates[i] - dates[start_idx]).days)
            start_idx = None

    if not durations:
        return None
    arr = np.asarray(durations, float)
    return {
        "cycles": int(arr.size),
        "rec_min": int(arr.min()),
        "rec_med": float(np.median(arr)),
        "rec_avg": round(float(arr.mean()), 1),
        "rec_max": int(arr.max()),
        "within_1w": round(float((arr <= 7).mean() * 100), 1),
        "over_2w": round(float((arr > 14).mean() * 100), 1),
        "below_now": start_idx is not None,   # 현재 미복귀(200일선 아래) 여부
    }


def beta_vs_index(close: pd.Series, index_close: pd.Series):
    """최근 BETA_YEARS 일간수익률 기준 베타·상관계수. 표본 부족 시 None."""
    if index_close is None or close is None:
        return None
    cutoff = close.index.max() - pd.Timedelta(days=int(365.25 * BETA_YEARS))
    a = close[close.index >= cutoff].pct_change().dropna()
    b = index_close[index_close.index >= cutoff].pct_change().dropna()
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    joined.columns = ["asset", "index"]
    if len(joined) < MIN_BETA_DAYS:
        return None
    var = float(joined["index"].var())
    if var <= 0:
        return None
    cov = float(joined["asset"].cov(joined["index"]))
    corr = float(joined["asset"].corr(joined["index"]))
    return {"beta": round(cov / var, 2),
            "corr": round(corr, 2) if corr == corr else None,
            "beta_days": int(len(joined))}


def fetch_health(ticker):
    """ROE / 부채비율 / 영업이익률 + 시가총액. 앱 _fmt_health 와 같은 기준으로 등급."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    def num(*keys):
        for k in keys:
            v = info.get(k)
            if isinstance(v, (int, float)) and v == v:
                return float(v)
        return None

    roe, dte, opm = num("returnOnEquity"), num("debtToEquity"), num("operatingMargins")
    mcap = num("marketCap")

    good = total = 0
    if roe is not None:
        total += 1
        good += 1 if roe * 100 >= 15 else 0
    if dte is not None:
        total += 1
        good += 1 if dte < 100 else 0
    if opm is not None:
        total += 1
        good += 1 if opm * 100 >= 20 else 0

    if total == 0:
        grade = "-"
    elif good >= 2:
        grade = "🟢 건전"
    elif good == 1:
        grade = "🟡 보통"
    else:
        grade = "🔴 주의"
    return {"roe": roe, "dte": dte, "opm": opm, "mcap": mcap,
            "good": good, "total": total, "grade": grade}


def main():
    try:
        with open("winzone_data.json", encoding="utf-8") as f:
            winzone = json.load(f)["data"]
    except Exception as exc:
        print(f"winzone_data.json 을 먼저 만들어 주세요: {exc}")
        return 1

    targets = [(tk, v["name"], v["market"]) for tk, v in winzone.items()
               if v.get("market") in ("US", "KR")]
    print(f"대상 {len(targets)}개 (미장/국장 대형주) · 지수 다운로드 중...")

    indices = {}
    for key, ix in INDEX_FOR.items():
        s = load_close(ix, start="2010-01-01")
        indices[key] = s
        print(f"  {key}: {ix} {'OK ' + str(len(s)) + '행' if s is not None else '실패'}")

    result, skipped = {}, []
    for i, (tk, name, market) in enumerate(targets, 1):
        close = load_close(tk)
        if close is None:
            skipped.append(tk)
            print(f"  [{i}/{len(targets)}] {tk} {name}: 가격 데이터 없음, 스킵")
            continue
        rec = recovery_stats(close)
        if rec is None:
            skipped.append(tk)
            print(f"  [{i}/{len(targets)}] {tk} {name}: 복귀 사이클 없음, 스킵")
            continue

        idx_key = "US" if market == "US" else ("KQ" if tk.endswith(".KQ") else "KS")
        bt = beta_vs_index(close, indices.get(idx_key))
        health = fetch_health(tk)

        entry = {"name": name, "market": market, "index": INDEX_FOR[idx_key],
                 "history_start": close.index[0].date().isoformat(),
                 **rec, "health": health}
        if bt:
            entry.update(bt)
        result[tk] = entry

        print(f"  [{i}/{len(targets)}] {tk} {name}: 사이클 {rec['cycles']}회 "
              f"(최단 {rec['rec_min']}/중간 {rec['rec_med']:.0f}/최장 {rec['rec_max']}일) · "
              f"베타 {entry.get('beta', '-')} · 재무 {health['grade']}")
        time.sleep(0.05)

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "sma": SMA_WINDOW,
            "cycle_unit": "calendar_days",
            "unresolved_excluded": True,
            "beta_years": BETA_YEARS,
            "beta_min_days": MIN_BETA_DAYS,
            "index_for": INDEX_FOR,
            "price_source": "Yahoo Finance auto_adjust=True",
            "health_rule": "ROE>=15% / 부채비율<100% / 영업이익률>=20% 중 충족 개수",
        },
        "data": result,
    }
    with open("recovery_data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\n완료: {len(result)}개 종목 -> recovery_data.json "
          f"(스킵 {len(skipped)}: {skipped or '없음'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
