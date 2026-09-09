"""공시 뻐꾸기통 사전계산 — 내부자 매수 + 추적 인물·기관 지분 변동 → insider_data.json

미국은 SEC EDGAR(무료, 키 불필요), 한국은 DART OpenAPI(키 필요)를 쓴다.
DART 키가 없으면 한국 부분만 건너뛰고 미국 결과는 정상 생성한다.

핵심 설계: Form 4 는 90%가 노이즈다
  대부분의 Form 4 는 주식보상 수령(A)·옵션 행사(M)·세금납부용 인도(F)라서
  "자기 돈으로 샀다"는 신호가 아니다. 진짜 신호는 공개시장 매수(P)뿐이고,
  그중에서도 CEO/CFO 의 대규모 매수와 여러 임원이 같은 시기에 사는
  클러스터 매수가 의미 있다고 알려져 있다. 그래서 P 코드·비파생·금액 하한으로
  먼저 걸러내고 남은 것에만 점수를 매긴다.

준수 사항
  SEC: User-Agent 에 연락처 필수, 초당 10요청 제한. 서드파티 스크래핑 사이트는
       쓰지 않고 SEC 원본만 사용한다.
  DART: 키당 일 20,000 요청 제한(공식 안내 기준).

실행
  python precompute_insider.py                      # 최근 3영업일 수집
  python precompute_insider.py --days 7             # 최근 7일
  python precompute_insider.py --resolve "THIEL"    # CIK 찾기
  python precompute_insider.py --probe-dart         # DART 응답 스키마 확인
  python precompute_insider.py --skip-form4         # 워치리스트만 갱신(빠름)

환경변수
  SEC_USER_AGENT   예: "200sma-screener you@example.com"  (SEC 요구사항)
  DART_API_KEY     https://opendart.fss.or.kr 에서 무료 발급
"""
import argparse
import gzip
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "insider_data.json")
WATCHLIST = os.path.join(HERE, "watchlist.json")

SEC_UA = os.environ.get("SEC_USER_AGENT", "200sma-screener research contact@example.com")
DART_KEY = os.environ.get("DART_API_KEY", "").strip()

SEC_SLEEP = 0.12          # 초당 10요청 제한 준수 (여유 포함)
DART_SLEEP = 0.06
KEEP_DAYS = 90            # 롤링 보관 기간
MIN_BUY_USD = 100_000     # 내부자 매수 최소 금액
MIN_BUY_KRW = 100_000_000
CLUSTER_WINDOW = 30       # 클러스터 판정 기간(일)

# Form 4 거래 코드. P 만 '자기 돈으로 공개시장 매수'다.
BUY_CODES = {"P"}
NOISE_CODES = {"A", "M", "F", "G", "C", "E", "H", "I", "L", "W", "Z", "J", "K", "U", "D"}


# ------------------------------------------------------------------
# HTTP
# ------------------------------------------------------------------
def _get(url, ua=SEC_UA, timeout=30, retries=3, sleep=SEC_SLEEP):
    """단순 GET. gzip 응답과 일시적 오류를 처리한다."""
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={
            "User-Agent": ua,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "*/*",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                raw = res.read()
                if res.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                time.sleep(sleep)
                return raw
        except urllib.error.HTTPError as exc:
            last = exc
            # SEC 아카이브는 아직 없는 경로에 403 을 주기도 한다(예: 당일 daily-index).
            # 재시도해도 의미가 없으므로 바로 없는 것으로 처리한다.
            if exc.code in (403, 404):
                return None
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"    [경고] 요청 실패 {url} — {last}")
    return None


def _get_json(url, **kw):
    raw = _get(url, **kw)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return None


# ------------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------------
def load_watchlist():
    with open(WATCHLIST, encoding="utf-8") as f:
        wl = json.load(f)
    return wl.get("us", []), wl.get("kr", [])


def business_days(days_back, skip_today=True):
    """주말을 제외한 최근 영업일 목록 (최신순).

    daily-index 는 해당일 접수가 마감된 뒤 공개되므로 기본적으로 당일은 건너뛴다.
    당일을 넣으면 매번 403 을 맞고 빈 결과가 된다.
    """
    d = datetime.now(timezone.utc).date()
    if skip_today:
        d -= timedelta(days=1)
    out = []
    while len(out) < days_back:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out


def norm(s):
    return re.sub(r"\s+", "", str(s or "")).lower()


def _txt(node, path, default=None):
    """Form 4 는 값이 <field><value>X</value></field> 로 감싸여 있다."""
    if node is None:
        return default
    el = node.find(path)
    if el is None:
        return default
    v = el.find("value")
    t = (v.text if v is not None else el.text) or ""
    t = t.strip()
    return t if t else default


def _flag(node, path):
    v = _txt(node, path)
    return str(v).strip().lower() in ("1", "true", "y", "yes")


def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------
# CIK 조회 (대상 추가할 때만 사용 — 40MB 파일이라 평소엔 안 받는다)
# ------------------------------------------------------------------
def resolve_cik(query):
    print(f"cik-lookup-data.txt 다운로드 중 (40MB, 대상 추가 시에만)…")
    raw = _get("https://www.sec.gov/Archives/edgar/cik-lookup-data.txt", timeout=180)
    if raw is None:
        print("실패")
        return
    q = query.strip().lower()
    hits = []
    for line in raw.decode("latin-1").splitlines():
        if q in line.lower():
            parts = line.rstrip(":").rsplit(":", 1)
            if len(parts) == 2:
                hits.append((parts[0], parts[1].zfill(10)))
    print(f"'{query}' 검색 결과 {len(hits)}건")
    for name, cik in hits[:40]:
        print(f"  {cik}  {name}")


# ------------------------------------------------------------------
# 미국: Form 4 내부자 매수
# ------------------------------------------------------------------
def form4_index(day):
    """해당 일자의 Form 4 접수 목록. (accession, cik, 회사명, 파일경로)"""
    qtr = (day.month - 1) // 3 + 1
    url = (f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{qtr}/"
           f"form.{day.strftime('%Y%m%d')}.idx")
    raw = _get(url)
    if raw is None:
        return []
    rows = []
    for line in raw.decode("latin-1").splitlines():
        if not line.startswith("4 "):
            continue
        # 고정폭 포맷: Form Type | Company Name | CIK | Date Filed | File Name
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 5:
            continue
        form, company, cik, filed, path = parts[0], parts[1], parts[2], parts[3], parts[-1]
        if form.strip() != "4":
            continue
        f = filed.strip()
        if len(f) == 8 and f.isdigit():      # 20260908 → 2026-09-08 로 통일
            f = f"{f[:4]}-{f[4:6]}-{f[6:]}"
        rows.append({"cik": cik.strip(), "company": company.strip(),
                     "filed": f, "path": path.strip()})
    return rows


def parse_form4(raw_text, filed_date, path):
    """Form 4 전체 제출문에서 공개시장 매수(P)만 뽑아낸다."""
    m = re.search(r"<ownershipDocument>.*?</ownershipDocument>", raw_text, re.S)
    if not m:
        return []
    try:
        doc = ET.fromstring(m.group(0))
    except ET.ParseError:
        return []

    issuer = doc.find("issuer")
    name = _txt(issuer, "issuerName") or ""
    ticker = (_txt(issuer, "issuerTradingSymbol") or "").upper()
    if not ticker or ticker in ("NONE", "N/A"):
        return []

    # 문서 단위 10b5-1 표시 (2022년 개정으로 추가)
    doc_10b5 = _flag(doc, "aff10b5One")

    owners = []
    for ro in doc.findall("reportingOwner"):
        oid = ro.find("reportingOwnerId")
        rel = ro.find("reportingOwnerRelationship")
        owners.append({
            "owner": _txt(oid, "rptOwnerName") or "",
            "owner_cik": (_txt(oid, "rptOwnerCik") or "").zfill(10),
            "is_director": _flag(rel, "isDirector"),
            "is_officer": _flag(rel, "isOfficer"),
            "is_ten_pct": _flag(rel, "isTenPercentOwner"),
            "title": _txt(rel, "officerTitle") or "",
        })
    if not owners:
        return []

    table = doc.find("nonDerivativeTable")
    if table is None:
        return []

    # 한 사람이 같은 날 여러 번 쪼개 사는 경우가 흔하다(예: 4거래로 220만 달러).
    # 건별로 내보내면 피드가 지저분하고 금액 점수도 낮게 나오므로
    # (보고자, 거래일) 단위로 합산한다. 단가는 금액가중평균을 쓴다.
    agg = {}
    for tr in table.findall("nonDerivativeTransaction"):
        coding = tr.find("transactionCoding")
        code = _txt(coding, "transactionCode")
        if code not in BUY_CODES:
            continue
        amounts = tr.find("transactionAmounts")
        if _txt(amounts, "transactionAcquiredDisposedCode") != "A":
            continue          # 취득만
        shares = _num(_txt(amounts, "transactionShares"))
        price = _num(_txt(amounts, "transactionPricePerShare"))
        if not shares or not price:
            continue          # 가격 0 은 보상성 취득
        post = _num(_txt(tr.find("postTransactionAmounts"),
                         "sharesOwnedFollowingTransaction"))
        tdate = _txt(tr, "transactionDate") or filed_date
        plan = doc_10b5 or _flag(coding, "aff10b5One")
        for o in owners:
            k = (o["owner_cik"], tdate)
            cur = agg.setdefault(k, {"shares": 0.0, "cost": 0.0, "post": post,
                                     "plan": plan, "trades": 0, "owner": o,
                                     "tdate": tdate})
            cur["shares"] += shares
            cur["cost"] += shares * price
            cur["trades"] += 1
            cur["plan"] = cur["plan"] or plan
            if post is not None:
                cur["post"] = post if cur["post"] is None else max(cur["post"], post)

    acc = re.search(r"(\d{10}-\d{2}-\d{6})", path)
    out = []
    for (owner_cik, tdate), a in agg.items():
        if a["cost"] < MIN_BUY_USD:
            continue
        out.append({
            "market": "US",
            "ticker": ticker,
            "company": name,
            "filed": filed_date,
            "trade_date": tdate,
            "code": "P",
            "shares": a["shares"],
            "price": round(a["cost"] / a["shares"], 4),
            "value": round(a["cost"], 2),
            "trades": a["trades"],
            "shares_after": a["post"],
            "plan_10b5_1": a["plan"],
            "accession": acc.group(1) if acc else "",
            "url": f"https://www.sec.gov/Archives/{path}",
            **a["owner"],
        })
    return out


def collect_us_insider(days_back, max_filings):
    print(f"[미국] Form 4 수집 — 최근 {days_back}영업일")
    buys, seen = [], 0
    for day in business_days(days_back):
        idx = form4_index(day)
        if not idx:
            print(f"  {day}  접수 없음 또는 인덱스 미공개")
            continue
        print(f"  {day}  Form 4 {len(idx)}건 확인 중…")
        got = 0
        for row in idx:
            if seen >= max_filings:
                print(f"  [중단] 상한 {max_filings}건 도달")
                break
            seen += 1
            raw = _get(f"https://www.sec.gov/Archives/{row['path']}")
            if raw is None:
                continue
            hits = parse_form4(raw.decode("latin-1"), row["filed"], row["path"])
            buys.extend(hits)
            got += len(hits)
        print(f"    → 매수 신호 {got}건 (누적 {len(buys)})")
        if seen >= max_filings:
            break
    return buys


def collapse_cofilers(buys):
    """하나의 거래를 여러 계열 법인이 각각 신고한 건을 한 줄로 합친다.

    예: GOLD 를 Tether Global Investment / TPM S.A. / Devasini 가 같은 날 같은 수량·
    같은 단가로 신고한다. 실제로는 한 번의 매수인데 3건으로 세면 클러스터 점수가
    부풀려지고 화면도 같은 줄이 반복된다. (거래일, 수량, 단가)가 같으면 같은
    경제적 거래로 보고 묶는다.
    """
    groups = {}
    for b in buys:
        k = (b["ticker"], b.get("trade_date"), round(b.get("shares") or 0, 4),
             round(b.get("price") or 0, 4))
        groups.setdefault(k, []).append(b)

    out = []
    for rows in groups.values():
        # 직위가 있는(=개인 임원) 신고를 대표로 쓴다. 없으면 첫 건.
        rows.sort(key=lambda r: (0 if (r.get("title") or "").strip() else 1,
                                 0 if r.get("is_officer") else 1))
        head = dict(rows[0])
        if len(rows) > 1:
            others = len({r["owner_cik"] for r in rows}) - 1
            if others > 0:
                head["owner"] = f"{head['owner']} 외 {others}"
            head["co_filers"] = len(rows)
            head["co_filer_names"] = [r["owner"] for r in rows]
            # 관계 플래그는 합집합으로 본다 (한 곳이라도 임원이면 임원 신고 포함)
            for f in ("is_director", "is_officer", "is_ten_pct"):
                head[f] = any(r.get(f) for r in rows)
        else:
            head["co_filers"] = 1
        out.append(head)
    return out


def add_clusters(buys):
    """같은 종목을 서로 다른 매수 건으로 여러 번 산 흔적을 표시.

    계열 법인 공동신고를 collapse_cofilers 로 묶은 뒤 세므로,
    '독립적인 매수 이벤트 수'에 가깝다.
    """
    by_ticker = {}
    for b in buys:
        by_ticker.setdefault(b["ticker"], []).append(b)
    for rows in by_ticker.values():
        events = {}
        for r in rows:
            k = (r.get("trade_date"), round(r.get("shares") or 0, 4),
                 round(r.get("price") or 0, 4))
            events[k] = r.get("trade_date")
        n = len(events)
        span_ok = False
        if n >= 2:
            ds = sorted(d for d in events.values() if d)
            try:
                span = (datetime.fromisoformat(ds[-1]) - datetime.fromisoformat(ds[0])).days
                span_ok = span <= CLUSTER_WINDOW
            except ValueError:
                span_ok = True
        for r in rows:
            r["cluster_n"] = n if span_ok else 1
    return buys


def score_buys(buys):
    """신호 점수 0~100. 배점 근거는 화면 설명과 맞춘다."""
    for b in buys:
        parts = {}
        # 금액 (로그 스케일: 10만 달러 0점 → 1000만 달러 이상 만점)
        v = max(b["value"], MIN_BUY_USD)
        import math
        ratio = math.log10(v / MIN_BUY_USD) / 2.0        # 100x = 1.0
        parts["금액"] = 35 * min(1.0, ratio)
        # 직위 — 표기가 회사마다 달라 폭넓게 매칭한다
        title = (b.get("title") or "").lower()
        if any(k in title for k in ("chief executive", "ceo")):
            parts["직위"] = 25.0
        elif any(k in title for k in ("chief financial", "cfo")):
            parts["직위"] = 22.0
        elif any(k in title for k in ("exec chairman", "executive chairman",
                                      "chairman", "founder", "president")):
            parts["직위"] = 20.0
        elif b.get("is_officer"):
            parts["직위"] = 15.0
        elif b.get("is_director"):
            parts["직위"] = 10.0
        elif b.get("is_ten_pct"):
            parts["직위"] = 8.0
        else:
            parts["직위"] = 5.0
        # 보유 증가율 (기존 보유 대비 얼마나 늘렸나)
        after, sh = b.get("shares_after"), b.get("shares")
        if after and sh and after > sh:
            inc = sh / (after - sh)
            parts["보유증가"] = 20 * min(1.0, inc / 0.5)   # 기존의 50% 이상 증가면 만점
        else:
            parts["보유증가"] = 20.0 if after and sh and after <= sh else 6.0
        # 클러스터
        cn = b.get("cluster_n", 1)
        parts["클러스터"] = 0.0 if cn < 2 else min(20.0, 10.0 * (cn - 1))
        b["score"] = round(sum(parts.values()), 1)
        b["score_parts"] = {k: round(x, 1) for k, x in parts.items()}
    buys.sort(key=lambda x: -x["score"])
    return buys


# ------------------------------------------------------------------
# 미국: 추적 인물·기관 공시
# ------------------------------------------------------------------
_FORM_ALIASES = {
    "SC 13D": ("SC 13D", "SCHEDULE 13D", "SC 13D/A", "SCHEDULE 13D/A"),
    "SC 13G": ("SC 13G", "SCHEDULE 13G", "SC 13G/A", "SCHEDULE 13G/A"),
    "13F-HR": ("13F-HR", "13F-HR/A"),
    "4": ("4", "4/A"),
}


def collect_us_watchlist(us_list, days_back):
    print(f"[미국] 추적 대상 {len(us_list)}건 공시 확인")
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days_back)).isoformat()
    out = []
    for w in us_list:
        cik = w["cik"].zfill(10)
        d = _get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
        if not d:
            print(f"  [경고] {w['label']} 조회 실패 (CIK {cik})")
            continue
        rec = (d.get("filings") or {}).get("recent") or {}
        forms = rec.get("form") or []
        wanted = set()
        for f in w.get("forms", []):
            wanted.update(_FORM_ALIASES.get(f, (f,)))
        hits = 0
        for i, form in enumerate(forms):
            fd = (rec.get("filingDate") or [])[i] if i < len(rec.get("filingDate", [])) else ""
            if fd < cutoff or form.upper() not in {x.upper() for x in wanted}:
                continue
            acc = (rec.get("accessionNumber") or [])[i]
            doc = (rec.get("primaryDocument") or [""] * len(forms))[i]
            base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}"
            out.append({
                "market": "US", "label": w["label"], "name": w["name"], "kind": w["kind"],
                "cik": cik, "form": form, "filed": fd, "accession": acc,
                "subject": (rec.get("items") or [""] * len(forms))[i] or "",
                "url": f"{base}/{doc}" if doc else f"{base}/{acc}-index.htm",
                "delayed": form.upper().startswith("13F"),
            })
            hits += 1
        print(f"  {w['label']:<22} {hits}건")
    out.sort(key=lambda x: x["filed"], reverse=True)
    return out


# ------------------------------------------------------------------
# 한국: DART
# ------------------------------------------------------------------
DART_BASE = "https://opendart.fss.or.kr/api"


def dart_get(endpoint, **params):
    if not DART_KEY:
        return None
    params["crtfc_key"] = DART_KEY
    url = f"{DART_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    return _get_json(url, ua="200sma-screener", sleep=DART_SLEEP)


def probe_dart():
    """DART 응답 스키마를 실제로 찍어본다. 필드명이 문서와 다를 때 빠르게 확인용."""
    if not DART_KEY:
        print("DART_API_KEY 환경변수가 없습니다. https://opendart.fss.or.kr 에서 무료 발급 후\n"
              "  export DART_API_KEY=발급받은키\n로 설정하세요.")
        return
    end = datetime.now(timezone.utc).date()
    bgn = end - timedelta(days=14)
    print("1) 공시검색 list.json (지분공시 D)")
    d = dart_get("list.json", bgn_de=bgn.strftime("%Y%m%d"), end_de=end.strftime("%Y%m%d"),
                 pblntf_ty="D", page_no=1, page_count=10)
    print("   status:", (d or {}).get("status"), (d or {}).get("message"))
    items = (d or {}).get("list") or []
    print("   건수:", len(items))
    if items:
        print("   응답 키:", sorted(items[0].keys()))
        for it in items[:5]:
            print(f"     {it.get('rcept_dt')} {it.get('corp_name')} | {it.get('report_nm')}")
        code = items[0].get("corp_code")
        for ep in ("elestock.json", "majorstock.json"):
            r = dart_get(ep, corp_code=code)
            lst = (r or {}).get("list") or []
            print(f"2) {ep} (corp_code={code}) status={(r or {}).get('status')} 건수={len(lst)}")
            if lst:
                print("   응답 키:", sorted(lst[0].keys()))
                print("   예시:", {k: lst[0].get(k) for k in list(lst[0])[:8]})


def collect_kr(kr_list, days_back):
    """지분공시를 훑어 내부자(임원·주요주주) 매수와 추적 대상 대량보유를 뽑는다.

    DART 는 날짜 범위로 공시 목록을 주지만 상세는 corp_code 단위라,
    목록에서 대상 회사를 추린 뒤 상세를 조회하는 2단 구조로 간다.
    """
    if not DART_KEY:
        print("[한국] DART_API_KEY 없음 — 건너뜁니다. "
              "https://opendart.fss.or.kr 에서 무료 발급 후 환경변수로 넣으면 자동 수집됩니다.")
        return [], [], "no_key"

    end = datetime.now(timezone.utc).date()
    bgn = end - timedelta(days=days_back)
    print(f"[한국] 지분공시 수집 {bgn}~{end}")

    filings, page = [], 1
    while page <= 20:
        d = dart_get("list.json", bgn_de=bgn.strftime("%Y%m%d"), end_de=end.strftime("%Y%m%d"),
                     pblntf_ty="D", page_no=page, page_count=100)
        if not d or d.get("status") != "000":
            msg = (d or {}).get("message", "응답 없음")
            if page == 1:
                print(f"  [실패] list.json status={(d or {}).get('status')} {msg}")
                return [], [], f"list_error:{(d or {}).get('status')}"
            break
        items = d.get("list") or []
        filings.extend(items)
        if page >= int(d.get("total_page") or 1):
            break
        page += 1
    print(f"  지분공시 {len(filings)}건")

    ele_names = ("임원", "주요주주")
    major_names = ("대량보유",)
    ele_targets = {f["corp_code"]: f for f in filings
                   if any(k in (f.get("report_nm") or "") for k in ele_names)}
    major_targets = {f["corp_code"]: f for f in filings
                     if any(k in (f.get("report_nm") or "") for k in major_names)}
    print(f"  임원·주요주주 보고 대상 {len(ele_targets)}사 / 대량보유 보고 대상 {len(major_targets)}사")

    watch = [(norm(w["match"]), w) for w in kr_list]

    def matched(name):
        n = norm(name)
        for key, w in watch:
            if key and key in n:
                return w
        return None

    insiders, majors = [], []
    for code, f in ele_targets.items():
        r = dart_get("elestock.json", corp_code=code)
        for it in (r or {}).get("list") or []:
            if (it.get("rcept_dt") or "") < bgn.strftime("%Y%m%d"):
                continue
            delta = _num(it.get("sp_stock_lmp_cnt"))          # 특정증권등 소유증감
            price = _num(it.get("sp_stock_lmp_unpr"))         # 취득·처분 단가
            if delta is None or delta <= 0:
                continue                                      # 매수만
            value = (delta * price) if price else None
            if value is not None and value < MIN_BUY_KRW:
                continue
            insiders.append({
                "market": "KR", "ticker": f.get("stock_code") or "",
                "company": it.get("corp_name") or f.get("corp_name"),
                "owner": it.get("repror") or "",
                "title": it.get("isu_exctv_ofcps") or "",
                "relation": it.get("isu_exctv_rgist_at") or "",
                "trade_date": it.get("rcept_dt") or "",
                "filed": it.get("rcept_dt") or "",
                "shares": delta, "price": price, "value": value,
                "shares_after": _num(it.get("sp_stock_lmp_irds_cnt")),
                "report": it.get("report_tp") or "",
                "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={f.get('rcept_no')}",
                "watch": (matched(it.get("repror") or "") or {}).get("label"),
            })
    for code, f in major_targets.items():
        r = dart_get("majorstock.json", corp_code=code)
        for it in (r or {}).get("list") or []:
            if (it.get("rcept_dt") or "") < bgn.strftime("%Y%m%d"):
                continue
            w = matched(it.get("repror") or "")
            if not w:
                continue                                      # 대량보유는 추적 대상만
            majors.append({
                "market": "KR", "ticker": f.get("stock_code") or "",
                "company": it.get("corp_name") or f.get("corp_name"),
                "reporter": it.get("repror") or "",
                "label": w.get("label"), "kind": w.get("kind"),
                "filed": it.get("rcept_dt") or "",
                "ratio": _num(it.get("stkrt")),
                "ratio_prev": _num(it.get("bfsler_stkrt")),
                "reason": it.get("report_resn") or "",
                "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={f.get('rcept_no')}",
            })
    print(f"  → 내부자 매수 {len(insiders)}건 / 추적 대상 대량보유 {len(majors)}건")
    return insiders, majors, "ok"


# ------------------------------------------------------------------
# 병합 (일간 실행 누적)
# ------------------------------------------------------------------
def merge(old, new, keys):
    """기존 항목과 합치고 중복을 제거한 뒤 보관 기간을 넘긴 건 버린다."""
    cut = (datetime.now(timezone.utc).date() - timedelta(days=KEEP_DAYS)).isoformat()
    cut_kr = cut.replace("-", "")
    seen, out = set(), []
    for row in list(new) + list(old or []):
        k = tuple(str(row.get(x, "")) for x in keys)
        if k in seen:
            continue
        seen.add(k)
        d = str(row.get("filed") or "")
        if len(d) == 8 and d.isdigit():
            if d < cut_kr:
                continue
        elif d and d < cut:
            continue
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3, help="며칠 전까지 수집 (기본 3영업일)")
    ap.add_argument("--max-form4", type=int, default=4000, help="Form 4 조회 상한")
    ap.add_argument("--skip-form4", action="store_true", help="워치리스트만 갱신")
    ap.add_argument("--resolve", type=str, default="", help="CIK 검색")
    ap.add_argument("--probe-dart", action="store_true", help="DART 응답 스키마 확인")
    ap.add_argument("--out", type=str, default=OUT_JSON)
    args = ap.parse_args()

    if args.resolve:
        resolve_cik(args.resolve)
        return 0
    if args.probe_dart:
        probe_dart()
        return 0

    if "example.com" in SEC_UA:
        print("[주의] SEC 는 User-Agent 에 실제 연락처를 요구합니다. "
              "SEC_USER_AGENT 환경변수를 설정하세요.\n"
              '  export SEC_USER_AGENT="200sma-screener your@email.com"')

    us_list, kr_list = load_watchlist()
    t0 = time.time()

    us_buys = [] if args.skip_form4 else collect_us_insider(args.days, args.max_form4)
    raw_n = len(us_buys)
    us_buys = collapse_cofilers(us_buys)
    if raw_n:
        print(f"  계열 공동신고 합산: {raw_n}건 → {len(us_buys)}건")
    us_buys = score_buys(add_clusters(us_buys))
    us_watch = collect_us_watchlist(us_list, max(args.days, 30))
    kr_ins, kr_major, kr_status = collect_kr(kr_list, max(args.days, 14))

    old = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = {}

    merged_us = merge(old.get("us_insider_buys"), us_buys,
                      ["ticker", "trade_date", "shares", "price"])
    # 병합 후 다시 합산·클러스터 계산 (이전 실행분과 겹칠 수 있다)
    merged_us = score_buys(add_clusters(collapse_cofilers(merged_us)))
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "days_collected": args.days,
            "keep_days": KEEP_DAYS,
            "min_buy_usd": MIN_BUY_USD,
            "min_buy_krw": MIN_BUY_KRW,
            "cluster_window_days": CLUSTER_WINDOW,
            "buy_codes": sorted(BUY_CODES),
            "kr_status": kr_status,
            "sources": {
                "us": "SEC EDGAR daily-index + Form 4 XML + submissions API",
                "kr": "DART OpenAPI list/elestock/majorstock",
            },
            "notes": [
                "Form 4 는 공개시장 매수(P)·비파생·취득만 집계합니다. 주식보상(A)·"
                "옵션행사(M)·세금납부 인도(F)는 제외했습니다.",
                "13F 는 분기 종료 후 45일까지 지연 공시라 현재 보유를 뜻하지 않습니다.",
                "일별 자기주식 매입 공시(Form SR)는 2023년 법원 판결로 무효화되어 "
                "존재하지 않습니다. 바이백은 분기 단위만 확인 가능합니다.",
            ],
        },
        "us_insider_buys": merged_us,
        "us_watchlist": merge(old.get("us_watchlist"), us_watch,
                              ["cik", "accession", "form"]),
        "kr_insider_buys": merge(old.get("kr_insider_buys"), kr_ins,
                                 ["ticker", "owner", "trade_date", "shares"]),
        "kr_major_holdings": merge(old.get("kr_major_holdings"), kr_major,
                                   ["ticker", "reporter", "filed", "ratio"]),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    size = os.path.getsize(args.out) / 1024
    print(f"\n완료 ({time.time() - t0:.0f}초, {size:.0f} KB → {args.out})")
    print(f"  미국 내부자 매수 {len(payload['us_insider_buys'])}건 "
          f"(이번 수집 {len(us_buys)}건)")
    print(f"  미국 추적 공시   {len(payload['us_watchlist'])}건")
    print(f"  한국 내부자 매수 {len(payload['kr_insider_buys'])}건  [{kr_status}]")
    print(f"  한국 대량보유     {len(payload['kr_major_holdings'])}건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
