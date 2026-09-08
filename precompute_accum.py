"""
적립(분할매수) 전용 백테스트 사전계산 → accum_data.json 생성

기존 winzone_data.json 의 승률은 '목표 +10% / 손절 -5% / 3개월 만기' 기준의
단발 트레이드 통계다. 손절하지 않고 내려갈수록 더 사는 적립 전략과는
측정 대상이 다르므로, 적립 행동을 그대로 시뮬레이션해 실제 수익률 분포를 낸다.

기존 데이터에 영향을 주지 않기 위한 원칙:
  · winzone_data.json / recovery_data.json 을 절대 수정하지 않는다 (읽기만).
  · 종목 유니버스를 새로 조회하지 않고 winzone_data.json 의 키를 상속한다.
    (FinanceDataReader 로 KRX 시총순을 다시 받으면 종목 구성이 달라져
     기존 탭들의 표시 종목이 바뀌기 때문이다.)
  · 결과는 별도 파일 accum_data.json 에만 쓴다.

시뮬레이션 규칙
  진입    종가가 200일선 -5% 아래로 내려간 날 에피소드 시작
  사다리  -5 / -10 / -15 / -20 / -25 / -30%  (200일선 대비, 각 구간 에피소드당 1회)
  비중    equal   = 1 : 1 : 1 : 1 : 1 : 1
          pyramid = 1 : 1 : 1.5 : 1.5 : 2 : 2   (하방증량)
  청산    sma     = 종가가 200일선 위로 복귀한 날
          target  = 종가가 그때까지의 평균단가 +10% 에 닿은 날
          hold3   = 진입 후 63거래일(약 3개월) 경과
          hold6   = 진입 후 126거래일(약 6개월) 경과
          hold12  = 진입 후 252거래일(약 12개월) 경과
  에피소드는 겹치지 않는다(청산 다음 날부터 재탐색). 데이터 끝까지 청산되지
  않은 마지막 구간은 통계에서 제외한다(unresolved_excluded).

라벨
  진입일 기준 50일선·20일선 위/아래를 기록해 조건별로 나눠 집계한다.
  기간은 진입일 2016-01-01 기준으로 early / late 로 나눠 검증에 쓴다.

실행
  python precompute_accum.py                 # 전수
  python precompute_accum.py --limit 3       # 앞 3종목만 (검증용)
  python precompute_accum.py --tickers AAPL,NVDA
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_JSON = os.path.join(HERE, "winzone_data.json")     # 읽기 전용
OUT_JSON = os.path.join(HERE, "accum_data.json")

SMA_LONG, SMA_MID, SMA_SHORT = 200, 50, 20
LEVELS = [-5.0, -10.0, -15.0, -20.0, -25.0, -30.0]
WEIGHTS = {
    "equal": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "pyramid": [1.0, 1.0, 1.5, 1.5, 2.0, 2.0],
}
TARGET_PCT = 10.0
# 보유 기간 청산 규칙: 규칙 이름 → 보유 거래일 수
HOLD_BARS_MAP = {"hold3": 63, "hold6": 126, "hold12": 252}
HOLD_BARS_MAX = max(HOLD_BARS_MAP.values())
EXIT_RULES = ["sma", "target"] + list(HOLD_BARS_MAP)
# sma·target 규칙의 대기 상한. 상한에 닿으면 그 시점 종가로 강제 청산한다.
# 상한 없이 '조건 달성까지 대기'로 두면 미달 구간이 통계에서 빠져 승률이
# 100%에 수렴하는 동어반복이 된다(도달했으니 이겼다). 실패를 포함시키려면
# 반드시 상한이 필요하다.
MAX_WAIT_BARS = 756          # 약 3년
PERIOD_SPLIT = "2016-01-01"
MIN_EPISODES = 3        # 이 미만이면 저장하지 않음
THIN_EPISODES = 10      # 이 미만이면 표본 얇음으로 표시


def load_universe():
    """종목 목록을 winzone_data.json 에서 상속 (새로 조회하지 않는다)."""
    with open(SRC_JSON, encoding="utf-8") as f:
        wz = json.load(f)
    items = [(tk, v.get("name", tk), v.get("market", ""))
             for tk, v in wz["data"].items()]
    items.sort(key=lambda x: (x[2], x[0]))
    return items, str(wz.get("meta", {}).get("generated_at", "unknown"))


def load_close(ticker):
    """상장 전체 기간 수정종가."""
    for kwargs in ({"period": "max"}, {"start": "2000-01-01"}):
        try:
            raw = yf.download(ticker, auto_adjust=True, progress=False, **kwargs)
        except Exception:
            raw = None
        if raw is None or len(raw) == 0:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if "Close" not in raw.columns:
            continue
        s = raw["Close"].dropna().sort_index()
        if len(s) > SMA_LONG + min(HOLD_BARS_MAP.values()):
            return s
    return None


def simulate(dates, close, ma_long, ma_mid, ma_short, weights, exit_rule):
    """적립 에피소드 시뮬레이션. 규칙별로 독립 실행한다.

    같은 진입 시점이라도 청산 규칙이 다르면 체결 구간도 달라지므로
    (예: target 이 먼저 닿으면 아래 구간을 못 산다) 규칙마다 따로 돌린다.
    """
    n = len(close)
    total_w = float(sum(weights))
    trigger = LEVELS[0] / 100.0
    episodes = []
    i = 0
    while i < n:
        if close[i] / ma_long[i] - 1 > trigger:
            i += 1
            continue

        start = i
        filled = [False] * len(LEVELS)
        cost = wsum = 0.0
        min_rel = 0.0
        exit_idx = None
        j = i
        while j < n:
            gap = close[j] / ma_long[j] - 1
            for k, lv in enumerate(LEVELS):
                if not filled[k] and gap <= lv / 100.0:
                    filled[k] = True
                    cost += close[j] * weights[k]
                    wsum += weights[k]
            if wsum > 0:
                avg = cost / wsum
                rel = close[j] / avg - 1
                if rel < min_rel:
                    min_rel = rel
                if exit_rule == "sma" and close[j] > ma_long[j]:
                    exit_idx, forced = j, False
                    break
                if exit_rule == "target" and close[j] >= avg * (1 + TARGET_PCT / 100.0):
                    exit_idx, forced = j, False
                    break
                if exit_rule in HOLD_BARS_MAP and (j - start) >= HOLD_BARS_MAP[exit_rule]:
                    exit_idx, forced = j, False
                    break
                if exit_rule in ("sma", "target") and (j - start) >= MAX_WAIT_BARS:
                    exit_idx, forced = j, True      # 상한 도달 → 강제 청산(손실 포함)
                    break
            j += 1

        if exit_idx is None:       # 데이터 끝까지 미청산 → 제외하고 종료
            break

        avg = cost / wsum
        episodes.append({
            "ret": (close[exit_idx] / avg - 1) * 100,
            "inv": wsum / total_w * 100,
            "fills": int(sum(filled)),
            "hold": int((dates[exit_idx] - dates[start]).days),
            "dd": min_rel * 100,
            "forced": forced,
            "start": str(dates[start])[:10],
            "ma_mid_above": bool(close[start] > ma_mid[start]),
            "ma_short_above": bool(close[start] > ma_short[start]),
        })
        i = exit_idx + 1
    return episodes


def summarize(eps):
    if not eps or len(eps) < MIN_EPISODES:
        return None
    r = np.array([e["ret"] for e in eps], float)
    return {
        "n": len(eps),
        "win": round(float((r > 0).mean() * 100), 1),
        "ret_med": round(float(np.median(r)), 2),
        "ret_avg": round(float(r.mean()), 2),
        "ret_p25": round(float(np.percentile(r, 25)), 2),
        "ret_p75": round(float(np.percentile(r, 75)), 2),
        "inv_med": round(float(np.median([e["inv"] for e in eps])), 1),
        "fills_med": round(float(np.median([e["fills"] for e in eps])), 1),
        "hold_med": int(np.median([e["hold"] for e in eps])),
        "dd_med": round(float(np.median([e["dd"] for e in eps])), 2),
        "forced_pct": round(float(np.mean([e["forced"] for e in eps]) * 100), 1),
        "thin": len(eps) < THIN_EPISODES,
    }


def analyze(ticker):
    close_s = load_close(ticker)
    if close_s is None:
        return None
    ma_l = close_s.rolling(SMA_LONG).mean()
    ma_m = close_s.rolling(SMA_MID).mean()
    ma_s = close_s.rolling(SMA_SHORT).mean()
    valid = ma_l.notna() & ma_m.notna() & ma_s.notna()
    if valid.sum() < min(HOLD_BARS_MAP.values()) + 50:
        return None

    dates = close_s.index[valid]        # DatetimeIndex 유지 (뺄셈 결과에 .days 필요)
    c = close_s[valid].to_numpy(float)
    l = ma_l[valid].to_numpy(float)
    m = ma_m[valid].to_numpy(float)
    s = ma_s[valid].to_numpy(float)

    out = {"combos": {}, "history_start": str(close_s.index[0])[:10]}
    primary = None
    for rule in EXIT_RULES:
        for wname, w in WEIGHTS.items():
            eps = simulate(dates, c, l, m, s, w, rule)
            st = summarize(eps)
            if st:
                out["combos"][f"{rule}|{wname}"] = st
            if rule == "sma" and wname == "equal":
                primary = eps

    if primary:
        out["by_ma50"] = {
            "above": summarize([e for e in primary if e["ma_mid_above"]]),
            "below": summarize([e for e in primary if not e["ma_mid_above"]]),
        }
        out["by_ma20"] = {
            "above": summarize([e for e in primary if e["ma_short_above"]]),
            "below": summarize([e for e in primary if not e["ma_short_above"]]),
        }
        out["by_period"] = {
            "early": summarize([e for e in primary if e["start"] < PERIOD_SPLIT]),
            "late": summarize([e for e in primary if e["start"] >= PERIOD_SPLIT]),
        }
        out["episodes_total"] = len(primary)
    return out if out["combos"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N종목만 처리 (검증용)")
    ap.add_argument("--tickers", type=str, default="", help="콤마 구분 티커만 처리")
    ap.add_argument("--out", type=str, default=OUT_JSON)
    args = ap.parse_args()

    universe, src_gen = load_universe()
    if args.tickers:
        want = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        universe = [u for u in universe if u[0].upper() in want]
    if args.limit:
        universe = universe[:args.limit]

    print(f"대상 {len(universe)}종목 (유니버스 상속: winzone_data.json@{src_gen[:10]})")
    data, failed = {}, []
    t0 = time.time()
    for i, (tk, name, market) in enumerate(universe, 1):
        try:
            res = analyze(tk)
        except Exception as exc:
            res = None
            print(f"  [{i}/{len(universe)}] {tk} 예외: {exc}")
        if not res:
            failed.append(tk)
            print(f"  [{i}/{len(universe)}] {tk} 건너뜀")
            continue
        res["name"], res["market"] = name, market
        data[tk] = res
        p = res["combos"].get("sma|equal") or {}
        print(f"  [{i}/{len(universe)}] {tk:<12} 에피소드 {p.get('n','-'):>4} · "
              f"승률 {p.get('win','-'):>5}% · 중앙수익 {p.get('ret_med','-'):>7}% · "
              f"투입률 {p.get('inv_med','-'):>5}%")

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "universe_from": f"winzone_data.json@{src_gen}",
            "sma": {"long": SMA_LONG, "mid": SMA_MID, "short": SMA_SHORT},
            "levels": LEVELS,
            "weights": WEIGHTS,
            "exit_rules": {
                "sma": f"종가가 200일선 위로 복귀 시 청산 (최대 {MAX_WAIT_BARS}거래일 대기 후 강제 청산)",
                "target": f"평균단가 +{TARGET_PCT:.0f}% 도달 시 청산 (최대 {MAX_WAIT_BARS}거래일 대기 후 강제 청산)",
                **{k: f"진입 후 {v}거래일 경과 시 청산" for k, v in HOLD_BARS_MAP.items()},
            },
            "hold_bars": HOLD_BARS_MAP,
            "max_wait_bars": MAX_WAIT_BARS,
            "episode_policy": "non_overlapping, unresolved_excluded, forced_exit_at_max_wait",
            "period_split": PERIOD_SPLIT,
            "min_episodes": MIN_EPISODES,
            "thin_episodes": THIN_EPISODES,
            "price_source": "Yahoo Finance auto_adjust=True",
            "note": "적립 전용 통계. winzone_data.json 의 단발 트레이드 승률과 정의가 다르다.",
        },
        "data": data,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    size = os.path.getsize(args.out) / 1024
    print(f"\n완료: {len(data)}종목 저장, 실패 {len(failed)}종목, "
          f"{time.time()-t0:.0f}초, {size:.0f} KB → {args.out}")
    if failed:
        print("실패:", ", ".join(failed[:20]) + (" ..." if len(failed) > 20 else ""))


if __name__ == "__main__":
    main()
