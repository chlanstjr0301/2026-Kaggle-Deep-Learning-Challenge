"""vLLM 생성 공통 래퍼. 체크포인트 재개를 지원한다.

재개가 되어야 잡을 걸어두고 잘 수 있고, D-0에 런이 죽어도 이어서 할 수 있다.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path


def load_done(ckpt: Path):
    """{id: [outputs]}, {id: [trunc...]} 형태로 이미 끝난 것을 복원."""
    done, trunc, logp = {}, {}, {}
    if ckpt.exists():
        with open(ckpt, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["id"]] = r["outputs"]
                    trunc[r["id"]] = r.get("trunc", [False] * len(r["outputs"]))
                    logp[r["id"]] = r.get("logp", [None] * len(r["outputs"]))
                except Exception:
                    continue                      # 중간에 잘린 마지막 줄 무시
    return done, trunc, logp


def generate(llm, sampling, rows, ckpt: Path, prompt_fn, chunk: int = 100, log_every: int = 1):
    """rows: [(id, question), ...] -> {id: [text, ...]}

    chunk 단위로 ckpt(jsonl)에 append 한다. 이미 있는 id는 건너뛴다.
    """
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done, trmap, lpmap = load_done(ckpt)
    todo = [(i, q) for i, q in rows if i not in done]
    print(f"[gen] 전체 {len(rows)} · 완료 {len(done)} · 남음 {len(todo)}", flush=True)

    t0, ntok, ntrunc, nsamp = time.time(), 0, 0, 0
    with open(ckpt, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), chunk):
            batch = todo[s : s + chunk]
            outs = llm.generate([prompt_fn(q) for _, q in batch], sampling)
            for (i, _), o in zip(batch, outs):
                texts = [c.text for c in o.outputs]
                # max_tokens에 걸려 잘린 응답은 \boxed{}가 없어 폴백으로 흘러간다.
                # 이 비율이 정확도를 직접 갉아먹으므로 반드시 센다.
                trunc = [c.finish_reason == "length" for c in o.outputs]
                # 평균 로그확률 = 모델 자신의 확신도. 학습 없이 쓸 수 있는 신호라
                # 검증기를 만들기 전에 이것부터 시험해 볼 가치가 있다.
                lps = [(c.cumulative_logprob / max(len(c.token_ids), 1))
                       if getattr(c, "cumulative_logprob", None) is not None else None
                       for c in o.outputs]
                ntok += sum(len(c.token_ids) for c in o.outputs)
                ntrunc += sum(trunc); nsamp += len(trunc)
                done[i] = texts; trmap[i] = trunc; lpmap[i] = lps
                f.write(json.dumps({"id": i, "outputs": texts, "trunc": trunc,
                                    "logp": lps}, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())        # 프로세스가 죽어도 남도록
            if (s // chunk) % log_every == 0:
                el = time.time() - t0
                pct = (s + len(batch)) / max(len(todo), 1) * 100
                print(f"[gen] {s+len(batch)}/{len(todo)} ({pct:.1f}%) "
                      f"| {ntok/max(el,1e-9):.0f} tok/s | 경과 {el/60:.1f}분", flush=True)

    el = time.time() - t0
    tr = ntrunc / max(nsamp, 1)
    print(f"[gen] 완료. 신규 {ntok:,} 토큰 / {el/60:.1f}분 = {ntok/max(el,1e-9):.0f} tok/s"
          f" | 잘림 {ntrunc}/{nsamp} = {tr:.1%}"
          + ("  <-- max_tokens를 올려야 한다" if tr > 0.05 else ""), flush=True)
    return done, {"new_tokens": ntok, "seconds": el, "tok_per_s": ntok / max(el, 1e-9),
                  "trunc_ratio": round(tr, 4), "n_samples": nsamp,
                  "_trunc": trmap, "_logp": lpmap}   # id별. JSON에 남기지 말 것.
