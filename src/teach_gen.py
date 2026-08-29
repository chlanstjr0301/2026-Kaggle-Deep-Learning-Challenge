"""교사 모델로 학생이 못 푼 문항의 풀이를 생성한다. (규정 5.3a: 학습 데이터 구축 목적 허용)

대상 선정은 RFT r1 의 문항별 정답 수를 그대로 쓴다 (out/pass_at_k_r1.json).
  0/16      미해결        -> 교사 데이터 가치 최대
  1-3/16    불안정 해결    -> 여전히 가치 큼 (단일 생성 정답률 6~19%)
  4-7/16    부분 숙달      -> 소량
  8+/16     안정 숙달      -> 생성하지 않는다. 대신 학생 자신의 풀이를 replay 로 쓴다

교사에게 정답을 알려주지 않는다. 그래야 gold 가 독립적인 필터로 남는다.

  export DEEPSEEK_API_KEY=sk-...
  # 파일럿 (약 $0.83)
  python3 src/teach_gen.py --plan unsolved:4,hard:4 --limit-per-bucket 150 \
      --out out/teach_pilot.jsonl
  # 본 생성 (약 $11)
  python3 src/teach_gen.py --plan unsolved:6,hard:4,medium:2 --out out/teach_r1.jsonl
"""
import argparse, json, os, sys, time, random, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd

PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY"),
    "nvidia":   ("https://integrate.api.nvidia.com/v1/chat/completions", "NVIDIA_API_KEY"),
}
API = PROVIDERS["deepseek"][0]
SYSTEM = "You are a careful mathematician. You always finish with a final integer answer."
USER = ("Solve the following math problem step by step. The answer is a single integer. "
        "Put your final answer in \\boxed{{}}.\n\n{q}")

# $ per 1M tokens (2026-08 기준, 결제 전 콘솔에서 재확인할 것)
PRICE = {"deepseek-v4-pro": (0.435, 0.87), "deepseek-v4-flash": (0.14, 0.28)}
PRICE.update({m: (0.0, 0.0) for m in (
    "deepseek-ai/deepseek-v4-flash-0731",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b")})

BUCKETS = {"unsolved": lambda n: n == 0,
           "hard":     lambda n: 1 <= n <= 3,
           "medium":   lambda n: 4 <= n <= 7,
           "easy":     lambda n: n >= 8}

_lock = threading.Lock()
_rl_lock = threading.Lock()
_next_slot = [0.0]
RPM = [35]          # 전역 분당 요청 상한. --rpm 으로 조정
def _throttle():
    """워커 수·재시도와 무관하게 전체 호출률을 RPM 이하로 강제한다"""
    with _rl_lock:
        t = max(time.time(), _next_slot[0])
        _next_slot[0] = t + 60.0 / RPM[0]
    d = t - time.time()
    if d > 0: time.sleep(d)
_stat = {"in": 0, "out": 0, "cache": 0, "err": 0, "done": 0}
import collections as _c
_codes = _c.Counter()


def call(model, q, temp, key, tries=12):
    body = {"model": model, "temperature": temp, "max_tokens": 4096,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER.format(q=q)}]}
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for t in range(tries):
        try:
            _throttle()
            r = requests.post(API, headers=h, json=body, timeout=180)
            if r.status_code == 200:
                d = r.json()
                u = d.get("usage", {})
                with _lock:
                    _stat["in"] += u.get("prompt_tokens", 0)
                    _stat["out"] += u.get("completion_tokens", 0)
                    _stat["cache"] += u.get("prompt_cache_hit_tokens", 0)
                return d["choices"][0]["message"]["content"]
            if r.status_code in (408, 409, 425, 429, 500, 502, 503, 504, 529):
                with _lock: _codes[r.status_code] += 1
                time.sleep(min(2 ** t, 5) + random.random()); continue
            with _lock:
                _stat["err"] += 1; _codes[r.status_code] += 1
                if _codes[r.status_code] == 1:      # 처음 한 번은 본문을 보여준다
                    print(f"\n  \033[1;31mHTTP {r.status_code}: {r.text[:200]}\033[0m", flush=True)
            return None
        except Exception:
            time.sleep(min(2 ** t, 30) + random.random())
    with _lock: _stat["err"] += 1
    return None


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--plan", default="unsolved:6,hard:4,medium:2",
                    help="구간:생성횟수 쉼표구분")
    ap.add_argument("--data", default="out/train_clean.csv")
    ap.add_argument("--solved", default="out/pass_at_k_r1.json")
    ap.add_argument("--provider", default="deepseek", choices=list(PROVIDERS))
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--limit-per-bucket", type=int, default=0, help="파일럿용")
    ap.add_argument("--shard", default="", help="i/n 형식. 문항을 n등분해 i번째만 처리")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rpm", type=int, default=35, help="전역 분당 요청 상한")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    global API
    API, _envname = PROVIDERS[a.provider]
    RPM[0] = a.rpm
    key = os.environ.get(_envname)
    if not key:
        sys.exit(f"{_envname} 환경변수가 없다")

    df = pd.read_csv(a.data)
    q = dict(zip(df["id"], df["question"]))
    gold = dict(zip(df["id"], df["answer"]))
    ns = {k: int(v) for k, v in json.loads(Path(a.solved).read_text()).items()}

    plan = {}
    for part in a.plan.split(","):
        b, k = part.split(":")
        if b.strip() not in BUCKETS: sys.exit(f"모르는 구간: {b}")
        plan[b.strip()] = int(k)

    done = set()
    if Path(a.out).exists():
        for line in open(a.out, encoding="utf-8"):
            try: done.add(json.loads(line)["id"])
            except Exception: pass

    tasks = []
    rng = random.Random(42)
    for b, k in plan.items():
        ids = [i for i in df["id"] if i in ns and BUCKETS[b](ns[i])]
        rng.shuffle(ids)
        if a.shard:
            _i, _n = (int(x) for x in a.shard.split("/"))
            ids = ids[_i::_n]
        if a.limit_per_bucket: ids = ids[:a.limit_per_bucket]
        n_new = sum(1 for i in ids if i not in done)
        print(f"  {b:<10} 문항 {len(ids):>6,} · 생성 {k}회 · 신규 {n_new:,}")
        tasks += [(i, b, k) for i in ids if i not in done]

    if not tasks:
        print("남은 작업 없음"); return
    pi, po = PRICE.get(a.model, (0.435, 0.87))
    est = (len(tasks) * sum(plan.values()) / max(len(plan), 1))
    print(f"\n총 문항 {len(tasks):,} · 예상 호출 {sum(k for _,_,k in tasks):,}건")
    print(f"모델 {a.model} (${pi}/${po} per 1M) · 워커 {a.workers}\n")

    f = open(a.out, "a", encoding="utf-8")
    t0 = time.time()

    def work(t):
        i, b, k = t
        outs = [call(a.model, str(q[i]), a.temp, key) for _ in range(k)]
        outs = [o for o in outs if o]
        rec = {"id": i, "bucket": b, "n_solved": ns[i], "answer": int(gold[i]),
               "outputs": outs}
        with _lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
            _stat["done"] += 1
            d = _stat["done"]
            if d % 25 == 0 or d == len(tasks):
                cost = _stat["in"] / 1e6 * pi + _stat["out"] / 1e6 * po
                el = time.time() - t0
                cd = (" · " + " ".join(f"{k}:{v}" for k, v in _codes.most_common(3))) if _codes else ""
                print(f"[teach] {d}/{len(tasks)} 문항{cd} · 입력 {_stat['in']:,} "
                      f"(캐시 {_stat['cache']:,}) · 출력 {_stat['out']:,} "
                      f"· \033[1;36m${cost:.2f}\033[0m · 오류 {_stat['err']} "
                      f"· {el/60:.1f}분", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, tasks))
    f.close()

    cost = _stat["in"] / 1e6 * pi + _stat["out"] / 1e6 * po
    n = max(_stat["done"], 1)
    print(f"\n완료: {a.out}")
    print(f"  실비 \033[1;36m${cost:.2f}\033[0m · 문항당 평균 출력 "
          f"{_stat['out']/n:.0f}토큰  ← 본 생성 예산은 이 값으로 다시 계산할 것")


if __name__ == "__main__":
    main()
