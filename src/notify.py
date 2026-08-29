"""텔레그램 알림. 잡이 끝나면 폰으로 결과를 보낸다.

설정 (인스턴스에서 한 번만):
  export TG_TOKEN='123456:AA...'      # @BotFather 가 준 토큰
  export TG_CHAT='987654321'          # 내 chat_id
  # 또는 ~/.tg 파일에 두 줄로:  TG_TOKEN=...  /  TG_CHAT=...

사용:
  python3 src/notify.py "RFT 끝남"
  python3 src/notify.py "밤 작업 완료" --log out/night.log --tail 25
  python3 src/notify.py "제출 파일" --file out/submission_maj64.csv
  python3 src/notify.py --summary                # 산출물 자동 요약해서 전송

설정이 없으면 조용히 화면에만 출력한다. 알림 때문에 잡이 죽으면 안 되므로
어떤 실패도 예외를 밖으로 내보내지 않는다.
"""
import argparse, json, os, re, sys, urllib.parse, urllib.request
from pathlib import Path

API = "https://api.telegram.org/bot{}/{}"
NOISE = re.compile(r"^(INFO|WARNING|\(EngineCore|Processed prompts|Rendering prompts|Loading safetensors|Capturing CUDA)")


def creds():
    tok, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    f = Path.home() / ".tg"
    if (not tok or not chat) and f.exists():
        for line in f.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"")
                if k == "TG_TOKEN": tok = tok or v
                if k == "TG_CHAT":  chat = chat or v
    return tok, chat


def post(method, fields, files=None):
    tok, chat = creds()
    if not tok or not chat:
        return False, "설정 없음 (TG_TOKEN / TG_CHAT)"
    fields = dict(fields, chat_id=chat)
    url = API.format(tok, method)
    try:
        if files:
            b = f"----{os.urandom(8).hex()}".encode()
            body = b""
            for k, v in fields.items():
                body += (b"--" + b + b"\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                         % (k.encode(), str(v).encode()))
            for k, p in files.items():
                data = Path(p).read_bytes()
                body += (b"--" + b + b"\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
                         b"Content-Type: application/octet-stream\r\n\r\n" % (k.encode(), Path(p).name.encode()))
                body += data + b"\r\n"
            body += b"--" + b + b"--\r\n"
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": f"multipart/form-data; boundary={b.decode()}"})
        else:
            req = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode())
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read()).get("ok", False), "전송됨"
    except Exception as e:                      # 알림 실패로 잡을 죽이지 않는다
        return False, f"{type(e).__name__}: {e}"


def send(text, file=None):
    text = text[:3900]
    if file and Path(file).exists() and Path(file).stat().st_size < 45 * 1024 * 1024:
        ok, msg = post("sendDocument", {"caption": text}, {"document": file})
    else:
        ok, msg = post("sendMessage", {"text": text, "disable_web_page_preview": "true"})
    print(f"[notify] {msg}\n{text}", file=sys.stderr)
    return ok


def log_tail(path, n):
    """진행바·INFO 로그를 걸러낸 마지막 n줄."""
    try:
        lines = [l.rstrip() for l in Path(path).read_text(errors="replace").splitlines()]
    except Exception:
        return ""
    keep = [l for l in lines if l.strip() and not NOISE.match(l.strip())]
    return "\n".join(keep[-n:])


def summary(od="out"):
    """산출물이 있으면 핵심 수치만 뽑아 한 화면으로."""
    od, out = Path(od), []
    for f in sorted(od.glob("rft_summary_*.json")):
        d = json.loads(f.read_text())
        out.append(f"[RFT {d.get('tag')}] 커버리지 {d.get('coverage')} · "
                   f"SFT {d.get('sft_samples'):,} · 오답 {d.get('negatives'):,} · "
                   f"라벨의심 {d.get('label_suspect')}")
    f = od / "diag_report.json"
    if f.exists():
        d = json.loads(f.read_text())
        c = d.get("curve", [])
        if c:
            out.append(f"[진단 k={d.get('k')}] " + " ".join(
                f"maj@{x['m']}={x['maj']}" for x in c[-3:])
                + f" · pass@{d.get('k')}={c[-1]['pass']}"
                + f" · 서로다른답 {d.get('uniq_answers')}")
    f = od / "baseline_report.json"
    if f.exists():
        d = json.loads(f.read_text())
        for k, v in d.items():
            out.append(f"[{k}] acc={v.get('acc')} 잘림={v.get('perf',{}).get('trunc_ratio')}")
    for f in sorted(od.glob("submission*.csv")):
        out.append(f"[제출] {f.name} ({sum(1 for _ in open(f))-1}행)")
    return "\n".join(out) or "산출물 없음"


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("text", nargs="?", default="알림")
    ap.add_argument("--log", default=None)
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--file", default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out-dir", default="out")
    a = ap.parse_args()

    parts = [a.text]
    if a.summary:
        parts.append("\n" + summary(a.out_dir))
    if a.log:
        t = log_tail(a.log, a.tail)
        if t:
            parts.append(f"\n--- {Path(a.log).name} ---\n{t}")
    send("\n".join(parts), a.file)


if __name__ == "__main__":
    main()
