"""D-7 / D-5: 거부 표집(RFT) 데이터 생성.

핵심 설계 — 난이도 역비례 배분. 문항당 개수를 균일하게 자르면
쉬운 문항이 학습 데이터를 잠식한다(논문 5.4절).

부산물 셋을 함께 저장한다:
  negatives.jsonl     오답 풀이       -> D-4 오답 재활용
  pass_at_k.json      문항별 성공 횟수 -> D-5 난이도 지도
  label_suspect.csv   만장일치 불일치  -> 라벨 오류 후보

  # 1라운드 (전체, k=16, T=0.8 — D-8 실측 설정)
  python3 src/rft_sample.py --model ./qwen3b --data out/train_clean.csv \
      --k 16 --temp 0.8 --tag r1 --quarantine out/train_quarantine.csv
  # 2라운드 (1라운드에서 0개 맞춘 문항만, 고온으로 탐색)
  python3 src/rft_sample.py --model ./qwen3b --data out/train_clean.csv \
      --k 32 --temp 1.2 --tag r2 --only-unsolved out/pass_at_k_r1.json
"""
import argparse, json, re, sys, collections
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from extract import extract_answer
from prompts import build_prompt, DEFAULT
from gen import generate


def model_tag(path: str) -> str:
    """모델 식별자. 캐시 키에 반드시 들어가야 한다 —— 이게 없으면 SFT 모델 평가가
    베이스 모델의 생성 결과를 조용히 재사용하고 '완료'라고 보고한다."""
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]", "_", Path(path).name.strip("/")) or "model"

MIN_CHARS = 100          # 이보다 짧으면 추론 없이 답만 찍은 것
PREFIX_FRAC = 0.25       # 정답이 앞 25% 안에 나오면 우연 일치 의심
PREFIX_MIN_DIGITS = 3    # 1~2자리 정답에는 이 검사를 쓰지 않는다 (아래 주석)


def kappa(n_solved: int, k: int) -> int:
    """난이도 역비례 배분: 쉬울수록 적게 가져간다.

    정답 '개수'가 아니라 '비율'로 판정한다. 개수 기준(>=7)을 쓰면 k를 8에서
    16으로 올렸을 때 같은 난이도의 문항이 갑자기 쉬운 쪽으로 분류돼
    배분이 통째로 틀어진다. 비율은 k에 불변이다.
    """
    if n_solved == 0:
        return 0
    r = n_solved / k
    if r >= 0.50: return 1               # 쉬움: 한 개만
    if r >= 0.25: return 2               # 보통
    return min(n_solved, 4)              # 어려움: 있는 대로 (최대 4)


def early_answer(text: str, g: int) -> bool:
    """정답이 풀이 앞부분에 나오면 우연 일치를 의심한다.

    단, 1~2자리 정답에는 적용하지 않는다. 학습셋 정답의 69.5%가 1~2자리인데
    "7" 같은 문자열은 어떤 풀이의 앞부분에도 우연히 등장한다. 그대로 두면
    멀쩡한 풀이를 대량으로 버리게 된다.
    """
    if len(str(abs(g))) < PREFIX_MIN_DIGITS:
        return False
    head = text[: int(len(text) * PREFIX_FRAC)]
    return re.search(rf"(?<![\d.]){re.escape(str(g))}(?![\d.])", head) is not None


def path_hash(text: str) -> str:
    """계산 경로 해시. 중간 산술식의 순서열로 중복 풀이를 제거한다."""
    ops = re.findall(r"-?\d[\d,]*(?:\.\d+)?\s*[-+*/×÷]\s*-?\d[\d,]*(?:\.\d+)?", text)
    return "|".join(o.replace(" ", "").replace(",", "") for o in ops) or text[:80]


def select(cands, keep: int):
    """중복 제거 후 길이 중앙값에 가까운 순으로 keep개.
    최단을 고르면 추론이 무너진다(논문 5.4절) — 중앙값을 쓴다."""
    seen, uniq = set(), []
    for t in cands:
        h = path_hash(t)
        if h in seen: continue
        seen.add(h); uniq.append(t)
    if not uniq: return []
    med = sorted(len(t) for t in uniq)[len(uniq) // 2]
    return sorted(uniq, key=lambda t: abs(len(t) - med))[:keep]


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="out/train_clean.csv")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--k", type=int, default=16)      # pass@16=0.90 (D-8 실측)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=3072)  # D-9 실측으로 확정
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--prompt", default=DEFAULT)
    ap.add_argument("--only-unsolved", default=None,
                    help="이전 라운드 pass_at_k.json — 성공 0인 문항만 대상")
    ap.add_argument("--consensus", type=float, default=1.0,
                    help="격리 판정에 요구할 합의 비율 (1.0=만장일치). "
                         "1.0은 어려운 문항을 전부 drop시키므로 0.85~0.95도 검토할 것")
    ap.add_argument("--quarantine", default=None,
                    help="격리 문항 CSV (current_label / suggested_answer). "
                         "만장일치 검사로 재라벨·유지·폐기를 판정한다")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data)

    if a.only_unsolved:
        prev = json.loads(Path(a.only_unsolved).read_text())
        hard = {i for i, n in prev.items() if n == 0}
        df = df[df["id"].isin(hard)].reset_index(drop=True)
        print(f"[rft] 어려운 구간만: {len(df)}문항")

    rows = list(zip(df["id"], df["question"]))
    gold = dict(zip(df["id"], df["answer"]))
    qtext = dict(zip(df["id"], df["question"]))

    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=4096, seed=42)
    sp = SamplingParams(n=a.k, temperature=a.temp, top_p=0.95,
                        max_tokens=a.max_tokens, seed=42)
    tok = llm.get_tokenizer()
    pf = lambda q: build_prompt(tok, q, a.prompt, chat=True)   # D-9: chat 우세 (p=0.0016)
    outs, perf = generate(llm, sp, rows, od / f"gen_rft_{model_tag(a.model)}__{a.tag}.jsonl", pf)

    sft, neg, passk, suspect = [], [], {}, []
    kappa_hist = collections.Counter()

    for i, _ in rows:
        texts = outs.get(i, [])
        preds = [extract_answer(t) for t in texts]
        g = int(gold[i])
        correct = [t for t, p in zip(texts, preds) if p == g]
        wrong   = [t for t, p in zip(texts, preds) if p != g]
        n = len(correct)
        passk[i] = n

        # 만장일치 불일치 -> 라벨 오류 후보
        nn = [p for p in preds if p is not None]
        if len(nn) == len(preds) and len(set(nn)) == 1 and nn[0] != g:
            suspect.append({"id": i, "label": g, "model_unanimous": nn[0],
                            "question": qtext[i][:300]})

        for t in wrong:
            neg.append({"id": i, "question": qtext[i], "solution": t, "label": 0})

        # 필터: 길이 하한 + 정답 조기 등장
        cand = [t for t in correct if len(t) >= MIN_CHARS and not early_answer(t, g)]
        keep = kappa(n, a.k)
        kappa_hist[keep] += 1
        for t in select(cand, keep):
            sft.append({"id": i, "question": qtext[i], "solution": t, "answer": g, "label": 1})

    def dump(obj, name):
        p = od / name
        with open(p, "w", encoding="utf-8") as f:
            if name.endswith(".jsonl"):
                for r in obj: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        return p

    dump(neg, f"negatives_{a.tag}.jsonl")
    dump(passk, f"pass_at_k_{a.tag}.json")
    pd.DataFrame(suspect).to_csv(od / f"label_suspect_{a.tag}.csv", index=False)
    # sft 는 격리 편입분을 합친 뒤에 쓴다 (아래 블록에서 sft.extend 가 일어난다).

    cov = sum(1 for v in passk.values() if v > 0) / max(len(passk), 1)
    summary = {"tag": a.tag, "k": a.k, "temp": a.temp, "n_problems": len(rows),
               "quarantine": a.quarantine,
               "coverage": round(cov, 4), "sft_samples": len(sft),
               "negatives": len(neg), "label_suspect": len(suspect),
               "kappa_hist": dict(kappa_hist), "perf": perf}
    dump(summary, f"rft_summary_{a.tag}.json")

    # ── 격리 문항 판정 ──────────────────────────────────────────
    # 라벨이 의심스러운 문항을 버리지 않고, 모델의 만장일치를 독립 심판으로 쓴다.
    # 운영진도 리더보드 검수에서 "정답이 명확히 확인되는 문항은 정답을 수정하여
    # 유지"했으므로 재라벨링은 승인된 처리다. 다만 제안값은 다른 참가자의
    # 계산이므로 모델이 독립적으로 같은 답에 도달할 때만 채택한다.
    if a.quarantine and Path(a.quarantine).exists():
        qdf = pd.read_csv(a.quarantine)
        qrows = list(zip(qdf["id"], qdf["question"]))
        qouts, _ = generate(llm, sp, qrows, od / f"gen_quar_{model_tag(a.model)}__{a.tag}.jsonl", pf)
        cur = dict(zip(qdf["id"], pd.to_numeric(qdf["current_label"], errors="coerce")))
        sug = dict(zip(qdf["id"], pd.to_numeric(qdf["suggested_answer"], errors="coerce")))
        qq  = dict(zip(qdf["id"], qdf["question"]))

        verdicts, adopted = [], []
        for i, _ in qrows:
            preds = [extract_answer(t) for t in qouts.get(i, [])]
            nn = [p for p in preds if p is not None]
            # 합의 비율 기준. 파싱된 표 중 최빈값의 점유율이 임계 이상이면 채택.
            unanimous = None
            if nn:
                top, cnt = collections.Counter(nn).most_common(1)[0]
                if cnt / len(preds) >= a.consensus:
                    unanimous = top
            c, sg = cur.get(i), sug.get(i)
            if unanimous is not None and sg == sg and unanimous == int(sg):
                v, lab = "relabel", int(sg)          # 제보값을 모델이 독립 확인
            elif unanimous is not None and c == c and unanimous == int(c):
                v, lab = "keep", int(c)              # 원 라벨이 맞았다
            else:
                v, lab = "drop", None                # 판정 불가
            verdicts.append({"id": i, "verdict": v, "current": c, "suggested": sg,
                             "model_unanimous": unanimous, "final": lab})
            if lab is not None:
                texts = [t for t, p in zip(qouts.get(i, []), preds) if p == lab]
                cand = [t for t in texts if len(t) >= MIN_CHARS and not early_answer(t, lab)]
                for t in select(cand, kappa(len(texts), a.k)):
                    adopted.append({"id": i, "question": qq[i], "solution": t,
                                    "answer": lab, "label": 1})

        vdf = pd.DataFrame(verdicts)
        vdf.to_csv(od / f"quarantine_verdict_{a.tag}.csv", index=False)
        sft.extend(adopted)
        vc = vdf["verdict"].value_counts().to_dict()
        print(f"\n== 격리 판정 ({len(qrows)}문항) ==")
        for k in ("relabel", "keep", "drop"):
            print(f"  {k:8s} {vc.get(k,0):4d}")
        print(f"  편입된 SFT 표본 {len(adopted)}")

    # 난이도 분포 — 이게 RFT 데이터의 성격을 결정한다
    rate = collections.Counter()
    for v in passk.values():
        r = v / a.k
        rate["못품(0)" if v == 0 else "어려움(<25%)" if r < .25
             else "보통(25~50%)" if r < .5 else "쉬움(>=50%)"] += 1
    N = max(len(passk), 1)

    dump(sft, f"sft_{a.tag}.jsonl")        # 격리 편입분 포함

    print("\n== 난이도 분포 ==")
    for k_ in ("못품(0)", "어려움(<25%)", "보통(25~50%)", "쉬움(>=50%)"):
        print(f"  {k_:<14} {rate[k_]:>6,}  ({rate[k_]/N:>5.1%})")
    print(f"  κ 배분: " + "  ".join(f"{k_}개→{v:,}문항" for k_, v in sorted(kappa_hist.items())))

    print("\n== 판정 게이트 ==")
    # D-8 실측: val pass@16 = 0.9008. train도 같은 분포이므로 여기서 크게 벗어나면
    # 생성 설정이 실측과 달라졌다는 뜻이다 (프롬프트·온도·길이 확인할 것).
    exp_lo, exp_hi = 0.85, 0.94
    print(f"  pass@{a.k} 커버리지  {cov:.3f}   기대 {exp_lo}~{exp_hi} (val 실측 0.901)  "
          f"{'OK' if exp_lo <= cov <= exp_hi else '!! 실측과 어긋남 — 생성 설정 확인'}")
    print(f"  라벨 의심          {len(suspect):,}   기준 <500  "
          f"{'OK' if len(suspect) < 500 else '!! 확인 필요'}")
    hard = rate["못품(0)"] + rate["어려움(<25%)"]
    print(f"  어려운 문항        {hard:,} ({hard/N:.1%})  ← 2라운드(고온) 대상")
    print(f"  SFT 표본          {len(sft):,}   문항당 평균 {len(sft)/N:.2f}개")
    print(f"  오답 표본          {len(neg):,}  (D-4 재활용용)")
    if len(sft) < 8000:
        print("  !! SFT 표본이 너무 적다. MIN_CHARS / early_answer 필터가 과하게 걸렸는지 확인.")


if __name__ == "__main__":
    main()
