"""사전계산 산출물이 커밋할 만한 상태인지 검증하는 게이트.

자동 갱신에서 가장 위험한 실패는 '에러 없이 나쁜 데이터를 덮어쓰는 것'이다.
yfinance 가 절반쯤 실패하면 스크립트는 정상 종료하지만 종목이 뭉텅이로 빠진
JSON 이 만들어진다. 그 상태로 커밋되면 앱에서 종목이 조용히 사라진다.

그래서 갱신 전 파일과 비교해 종목 수가 크게 줄거나 필수 키가 빠지면
0 이 아닌 코드로 종료해 커밋을 막는다.

실행:
  python validate_precompute.py --old /tmp/recovery_data.json \
                                --new recovery_data.json --kind recovery
  python validate_precompute.py --old /tmp/accum_data.json \
                                --new accum_data.json --kind accum
"""
import argparse
import json
import sys

MIN_KEEP_RATIO = 0.90      # 기존 종목 수의 이 비율 미만이면 실패
MIN_FILLED_RATIO = 0.80    # 통계가 실제로 담긴 종목 비율 하한

REQUIRED = {
    "recovery": ("cycles", "rec_med", "rec_avg", "rec_max", "within_1w", "over_2w"),
    "accum": ("combos", "history_start"),
}


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or "data" not in doc or "meta" not in doc:
        raise ValueError(f"{path}: meta/data 구조가 아닙니다")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="갱신 전 파일 (없으면 비교를 건너뜀)")
    ap.add_argument("--new", required=True, help="갱신 후 파일")
    ap.add_argument("--kind", required=True, choices=sorted(REQUIRED))
    args = ap.parse_args()

    fails, notes = [], []

    try:
        new = load(args.new)
    except Exception as exc:
        print(f"❌ 새 파일을 읽을 수 없습니다: {exc}")
        return 1

    nd = new["data"]
    notes.append(f"새 파일 종목 수: {len(nd)}")
    if not nd:
        fails.append("새 파일에 종목이 없습니다")

    # 필수 키 확인
    req = REQUIRED[args.kind]
    missing = [tk for tk, v in nd.items()
               if not isinstance(v, dict) or any(k not in v for k in req)]
    if missing:
        fails.append(f"필수 키 누락 {len(missing)}종목 (예: {missing[:5]})")

    # 통계가 실제로 담겼는지 (accum 은 combos 가 비어 있을 수 있다)
    if args.kind == "accum":
        filled = sum(1 for v in nd.values() if v.get("combos"))
        ratio = filled / max(1, len(nd))
        notes.append(f"combos 보유 종목: {filled}/{len(nd)} ({ratio:.0%})")
        if ratio < MIN_FILLED_RATIO:
            fails.append(f"combos 보유 비율 {ratio:.0%} < {MIN_FILLED_RATIO:.0%}")

    # 갱신 전과 비교
    try:
        old = load(args.old)
    except Exception as exc:
        notes.append(f"비교 생략 (기존 파일 없음/읽기 실패: {exc})")
    else:
        od = old["data"]
        keep = len(nd) / max(1, len(od))
        notes.append(f"기존 {len(od)}종목 → 새 {len(nd)}종목 (유지율 {keep:.0%})")
        if keep < MIN_KEEP_RATIO:
            fails.append(f"종목 유지율 {keep:.0%} < {MIN_KEEP_RATIO:.0%} — "
                         "데이터 제공처 대량 실패로 보입니다")
        gone = [tk for tk in od if tk not in nd]
        if gone:
            notes.append(f"사라진 종목 {len(gone)}개: {gone[:10]}")

    print(f"── {args.kind} 검증 ──")
    for n in notes:
        print(f"  · {n}")
    if fails:
        print("❌ 실패")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("✅ 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
