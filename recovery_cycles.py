"""
200일선 이탈→복귀 사이클 전수 산출 (로컬 검증용 스크립트)

앱에 내장된 `_RECOVERY_TOP50` / `_RECOVERY_DIST` 값이 실제 가격 데이터로
재현되는지 확인하기 위한 도구다. 앱 실행에는 사용되지 않는다.

사이클 정의:
  - 종가가 200일선(SMA200) 아래로 내려간 날 = 이탈(사이클 시작)
  - 이후 종가가 다시 200일선 위로 올라온 날 = 복귀(사이클 종료)
  - 아직 복귀하지 않은 마지막 구간은 제외(진행 중)

기간 단위는 두 가지를 모두 계산한다:
  - trading: 거래일 수 (복귀일 인덱스 - 이탈일 인덱스)
  - calendar: 달력 일수 (복귀일 날짜 - 이탈일 날짜)

2026-09 실행 결과, 원문 전수조사 값은 **달력일 기준**과 일치했다
(전체 평균 28.2/중앙값 5/p75 20/p90 79/p95 152, 분포 각 구간 편차 +6회 이내).
거래일 기준은 평균 19.4/중앙값 3으로 원문과 맞지 않는다.

실행: python recovery_cycles.py
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

SMA_WINDOW = 200

# 미국 시총 TOP50 (원문 전수조사 대상과 동일)
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "LLY", "AVGO",
    "JPM", "V", "UNH", "XOM", "MA", "JNJ", "PG", "COST", "HD", "ABBV",
    "WMT", "NFLX", "CRM", "BAC", "ORCL", "CVX", "MRK", "KO", "PEP", "AMD",
    "TMO", "ADBE", "CSCO", "LIN", "ACN", "MCD", "ABT", "WFC", "PM", "TXN",
    "DHR", "ISRG", "MS", "NEE", "QCOM", "INTC", "AMGN", "INTU", "RTX", "GS",
]

# 원문 전수조사 값: 티커 -> (사이클수, 평균, 중앙값, 최대)
ARTICLE = {
    "AAPL": (132, 40, 6, 333), "MSFT": (153, 26, 5, 399), "NVDA": (61, 45, 5, 444),
    "AMZN": (111, 30, 5, 607), "GOOGL": (67, 28, 4, 313), "META": (34, 32, 6, 392),
    "BRK-B": (122, 23, 4, 289), "TSLA": (83, 24, 7, 250), "LLY": (210, 31, 7, 592),
    "AVGO": (78, 12, 4, 166), "JPM": (203, 28, 6, 475), "V": (81, 14, 4, 164),
    "UNH": (151, 29, 5, 602), "XOM": (260, 27, 5, 497), "MA": (61, 21, 5, 238),
    "JNJ": (252, 26, 5, 375), "PG": (266, 25, 6, 387), "COST": (186, 22, 5, 337),
    "HD": (159, 29, 5, 329), "ABBV": (65, 20, 4, 406), "WMT": (219, 24, 5, 294),
    "NFLX": (52, 47, 6, 440), "CRM": (71, 28, 7, 387), "BAC": (226, 32, 5, 395),
    "ORCL": (163, 26, 6, 430), "CVX": (347, 21, 5, 429), "MRK": (227, 32, 6, 554),
    "KO": (240, 27, 6, 450), "PEP": (211, 25, 6, 433), "AMD": (151, 52, 6, 537),
    "TMO": (178, 29, 4, 515), "ADBE": (158, 27, 4, 406), "CSCO": (85, 48, 6, 418),
    "LIN": (142, 19, 4, 255), "ACN": (106, 24, 5, 302), "MCD": (251, 25, 5, 325),
    "ABT": (174, 26, 6, 316), "WFC": (243, 25, 5, 369), "PM": (88, 20, 6, 266),
    "TXN": (237, 33, 6, 463), "DHR": (168, 27, 4, 498), "ISRG": (101, 25, 5, 317),
    "MS": (127, 34, 6, 333), "NEE": (237, 20, 4, 420), "QCOM": (159, 33, 6, 537),
    "INTC": (150, 45, 7, 416), "AMGN": (220, 22, 5, 508), "INTU": (131, 26, 5, 521),
    "RTX": (213, 34, 6, 453), "GS": (90, 40, 6, 353),
}

# 원문 분포: 라벨 -> (하한, 상한, 횟수, 비율%)  (상한 None = 무제한)
ARTICLE_DIST = [
    ("1일", 1, 1, 1768, 22.4),
    ("2~3일", 2, 3, 1338, 16.9),
    ("1주 이내", 4, 7, 1580, 20.0),
    ("1~2주", 8, 14, 933, 11.8),
    ("2주~1개월", 15, 30, 743, 9.4),
    ("1~2개월", 31, 60, 554, 7.0),
    ("2~3개월", 61, 90, 297, 3.8),
    ("3~6개월", 91, 180, 371, 4.7),
    ("6개월~1년", 181, 365, 267, 3.4),
    ("1~2년", 366, 730, 49, 0.6),
    ("2년 이상", 731, None, 0, 0.0),
]


def load_close(ticker):
    """전체 기간 수정종가 시리즈. 실패 시 None."""
    for kwargs in ({"period": "max"}, {"start": "1900-01-01"}):
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
        if len(s) > SMA_WINDOW:
            return s
    return None


def extract_cycles(close: pd.Series):
    """이탈→복귀 사이클 목록 반환.

    각 항목: dict(start, end, trading, calendar)
    미복귀(진행 중) 구간은 제외한다.
    """
    sma = close.rolling(SMA_WINDOW).mean()
    valid = sma.notna()
    px = close[valid].to_numpy(dtype=float)
    ma = sma[valid].to_numpy(dtype=float)
    dates = close.index[valid]

    below = px < ma
    above = px > ma

    cycles = []
    start_idx = None
    for i in range(len(px)):
        if start_idx is None:
            # 이탈 시작: 200일선 아래로 내려간 첫날
            if below[i]:
                start_idx = i
        elif above[i]:
            # 복귀 완료
            cycles.append({
                "start": dates[start_idx],
                "end": dates[i],
                "trading": i - start_idx,
                "calendar": (dates[i] - dates[start_idx]).days,
            })
            start_idx = None
    # start_idx 가 남아 있으면 아직 미복귀 → 제외
    unresolved = start_idx is not None
    return cycles, unresolved


def bucketize(durations):
    """원문 분포 구간별 횟수."""
    counts = []
    for label, lo, hi, _, _ in ARTICLE_DIST:
        if hi is None:
            n = int(sum(1 for d in durations if d >= lo))
        else:
            n = int(sum(1 for d in durations if lo <= d <= hi))
        counts.append((label, n))
    return counts


def summarize(durations):
    arr = np.asarray(durations, dtype=float)
    return {
        "cycles": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "min": int(arr.min()) if arr.size else 0,
        "max": int(arr.max()) if arr.size else 0,
        "p25": float(np.percentile(arr, 25)) if arr.size else float("nan"),
        "p75": float(np.percentile(arr, 75)) if arr.size else float("nan"),
        "p90": float(np.percentile(arr, 90)) if arr.size else float("nan"),
        "p95": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "within_1w": float((arr <= 7).mean() * 100) if arr.size else float("nan"),
        "over_2w": float((arr > 14).mean() * 100) if arr.size else float("nan"),
    }


def main():
    rows = []
    all_durations = {"trading": [], "calendar": []}
    failed = []

    for i, tk in enumerate(TICKERS, 1):
        close = load_close(tk)
        if close is None:
            failed.append(tk)
            print(f"  [{i}/{len(TICKERS)}] {tk}: 데이터 없음, 스킵")
            continue
        cycles, unresolved = extract_cycles(close)
        if not cycles:
            failed.append(tk)
            print(f"  [{i}/{len(TICKERS)}] {tk}: 사이클 없음, 스킵")
            continue

        trading = [c["trading"] for c in cycles]
        calendar = [c["calendar"] for c in cycles]
        all_durations["trading"].extend(trading)
        all_durations["calendar"].extend(calendar)

        t = summarize(trading)
        c = summarize(calendar)
        longest = max(cycles, key=lambda x: x["trading"])
        rows.append({
            "ticker": tk,
            "history_start": close.index[0].date().isoformat(),
            "cycles": t["cycles"],
            "unresolved_now": unresolved,
            "t_mean": round(t["mean"], 1), "t_median": t["median"], "t_max": t["max"],
            "c_mean": round(c["mean"], 1), "c_median": c["median"], "c_max": c["max"],
            "within_1w_trading_%": round(t["within_1w"], 1),
            "over_2w_trading_%": round(t["over_2w"], 1),
            "longest_start": longest["start"].date().isoformat(),
            "longest_end": longest["end"].date().isoformat(),
            "art_cycles": ARTICLE[tk][0], "art_mean": ARTICLE[tk][1],
            "art_median": ARTICLE[tk][2], "art_max": ARTICLE[tk][3],
        })
        print(f"  [{i}/{len(TICKERS)}] {tk}: {t['cycles']}회 "
              f"(거래일 평균 {t['mean']:.0f}/중앙값 {t['median']:.0f}/최대 {t['max']} · "
              f"달력 최대 {c['max']}) vs 원문 {ARTICLE[tk][0]}회 "
              f"평균 {ARTICLE[tk][1]}/중앙값 {ARTICLE[tk][2]}/최대 {ARTICLE[tk][3]}")

    if not rows:
        print("재현 실패: 데이터를 불러오지 못했습니다.")
        return 1

    df = pd.DataFrame(rows)
    df.to_csv("recovery_cycles_results.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"대상 {len(df)}개 종목 (실패 {len(failed)}: {failed or '없음'})")
    print(f"총 사이클: 거래일 기준 {len(all_durations['trading']):,}회 (원문 7,900회)")

    for unit in ("trading", "calendar"):
        s = summarize(all_durations[unit])
        label = "거래일" if unit == "trading" else "달력일"
        print(f"\n[{label} 기준 전체 통계]  원문: 평균 28 / 중앙값 5 / 최대 607 / "
              f"p25 2 / p75 20 / p90 79 / p95 152")
        print(f"  평균 {s['mean']:.1f} · 중앙값 {s['median']:.0f} · 최소 {s['min']} · 최대 {s['max']} · "
              f"p25 {s['p25']:.0f} · p75 {s['p75']:.0f} · p90 {s['p90']:.0f} · p95 {s['p95']:.0f}")
        print(f"  1주 이내 복귀 {s['within_1w']:.1f}% · 2주 초과 체류 {s['over_2w']:.1f}%")

        total = len(all_durations[unit])
        print(f"  {'구간':<12}{'재현':>8}{'비율':>8}{'원문':>8}{'원문비율':>9}{'차이':>8}")
        cum = 0
        for (label_b, n), (_, _, _, art_n, art_pct) in zip(bucketize(all_durations[unit]), ARTICLE_DIST):
            pct = n / total * 100 if total else 0
            cum += pct
            print(f"  {label_b:<12}{n:>8,}{pct:>7.1f}%{art_n:>8,}{art_pct:>8.1f}%{n - art_n:>+8,}")
        print(f"  누적 {cum:.1f}%")

    # 종목별 재현 정확도 (원문 단위인 달력일 기준으로 비교)
    n = len(df)
    df["cycles_diff"] = df["cycles"] - df["art_cycles"]
    df["mean_diff"] = df["c_mean"] - df["art_mean"]
    df["median_diff"] = df["c_median"] - df["art_median"]
    df["max_diff"] = df["c_max"] - df["art_max"]
    print("\n[종목별 재현 (달력일 기준 = 원문 단위)]")
    print(f"  최대체류 정확히 일치: {int((df['max_diff'] == 0).sum())}/{n}")
    print(f"  중앙값 정확히 일치  : {int((df['median_diff'].abs() < 0.01).sum())}/{n} "
          f"(±1일 이내 {int((df['median_diff'].abs() <= 1).sum())}/{n})")
    print(f"  평균 ±1일 이내      : {int((df['mean_diff'].abs() <= 1).sum())}/{n} "
          f"(±2일 {int((df['mean_diff'].abs() <= 2).sum())}/{n})")
    print(f"  사이클수 정확히 일치: {int((df['cycles_diff'] == 0).sum())}/{n} "
          f"(합계 {int(df['cycles_diff'].sum()):+d}회 — 원문 기준일 이후 신규 사이클 반영분)")

    diverged = df[(df["max_diff"] != 0) | (df["mean_diff"].abs() > 2)]
    if len(diverged):
        print("  차이가 큰 종목 (원문 기준일 이후 사이클 추가 여부 확인):")
        for _, r in diverged.iterrows():
            print(f"    {r['ticker']:6s} 평균 {r['c_mean']:.1f} vs {r['art_mean']} · "
                  f"최대 {r['c_max']} vs {r['art_max']} · "
                  f"최장 구간 {r['longest_start']}~{r['longest_end']}")

    print("\n[모을 시간 지표] 2주 초과 체류 비율 (중앙값보다 종목 구분력이 큼)")
    ranked = df.sort_values("over_2w_trading_%", ascending=False)
    for _, r in pd.concat([ranked.head(5), ranked.tail(5)]).iterrows():
        print(f"    {r['ticker']:6s} 2주 초과 {r['over_2w_trading_%']:>5.1f}% · "
              f"1주 이내 {r['within_1w_trading_%']:>5.1f}% · "
              f"평균 {r['c_mean']:>5.1f}일 · 중앙값 {r['c_median']:.0f}일")

    print("\n결과 CSV: recovery_cycles_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
