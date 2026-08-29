"""모델 출력 -> 정수 답 추출.

우선순위:
  1) 마지막 \boxed{...} / \fbox{...}   (중괄호 균형 파싱)
  2) 종결 표현 뒤의 첫 수
  3) 마지막 줄의 마지막 수
  4) 실패 -> None

원칙: 정확히 정수일 때만 채택한다. 반올림하지 않는다.
반올림을 허용하면 틀린 추론이 우연히 정답과 일치하는 경우가 생기고,
그것이 RFT 정답 필터를 오염시킨다.
"""
from __future__ import annotations
import re
from fractions import Fraction

TOL = 1e-9

# 단위·수식 잡음 (Numina 목록 기반, 이 대회 데이터에 맞춰 확장)
UNIT_WORDS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "inch", "ft", "feet",
    "hours", "hour", "minutes", "minute", "seconds", "km", "cm", "mm", "meters", "meter",
    "units", "unit", "points", "point", "digits", "digit", "cents", "students",
    "people", "years", "year", "days", "day", "times", "degrees", "degree",
]
_UNIT_RE = re.compile(r"\b(" + "|".join(UNIT_WORDS) + r")\b", re.I)

_CLOSERS = [
    r"final answer is", r"the answer is", r"answer\s*[:=]", r"answer is",
    r"therefore,?", r"thus,?", r"hence,?", r"so,?\s*the",
]
_CLOSER_RE = re.compile("(?:" + "|".join(_CLOSERS) + ")", re.I)

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _find_boxed(text: str) -> str | None:
    """마지막 \boxed{...} 내용. 중괄호를 세어 짝을 찾는다."""
    for tok in (r"\boxed", r"\fbox"):
        idx = text.rfind(tok)
        if idx == -1:
            continue
        i = text.find("{", idx)
        if i == -1:
            continue
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1 : j]
            j += 1
        return text[i + 1 :]          # 닫히지 않은 경우 끝까지
    return None


def _strip_latex(s: str) -> str:
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\(?:mbox|text|mathrm|textbf|mathbf)\s*\{[^}]*\}", "", s)
    s = re.sub(r"\\displaystyle|\;|\$|\\%|%", "", s)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")   # 유니코드 음부호
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = _UNIT_RE.sub("", s)
    return s.strip()


# 파싱은 됐지만 정수가 아님 -> 확정 폐기 (폴백으로 흘려보내면 안 된다)
class _NotInteger:
    _inst = None
    def __new__(cls):
        if cls._inst is None: cls._inst = super().__new__(cls)
        return cls._inst
NOT_INT = _NotInteger()

_SCI_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*\*\s*10\^\{?(-?\d+)\}?$")
_UNIT_EXP_RE = re.compile(r"\^\{?\d+\}?$")


def _frac_to_value(s: str):
    r"""\frac{a}{b}, \dfrac{a}{b}, a/b -> Fraction. 실패 시 None."""
    m = re.fullmatch(r"\\[dt]?frac\s*\{(-?\d+)\}\s*\{(-?\d+)\}", s.strip())
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return None if den == 0 else Fraction(num, den)
    m = re.fullmatch(r"\\[dt]?frac(-?\d)(-?\d)", s.strip())   # \frac23 축약형
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s.strip())
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return None if den == 0 else Fraction(num, den)
    return None


def _to_int(s):
    """문자열 -> 정수 / None(파싱 실패) / NOT_INT(파싱됐으나 비정수)."""
    if s is None:
        return None
    s = _strip_latex(str(s))
    if not s:
        return None
    s2 = s.replace(",", "").replace(" ", "")

    # 1) 분수: 파싱되면 정수 여부로 확정 판단
    fr = _frac_to_value(s2)
    if fr is not None:
        return int(fr) if fr.denominator == 1 else NOT_INT

    # 2) LaTeX 과학표기 a x 10^b  (단위 지수 제거보다 먼저)
    m = _SCI_RE.match(s2)
    if m:
        try:
            v = float(m.group(1)) * (10 ** int(m.group(2)))
        except (ValueError, OverflowError):
            return None
        r = round(v)
        return int(r) if abs(v - r) < max(TOL, abs(v) * 1e-12) else NOT_INT

    # 3) 단위 지수 꼬리 제거 (cm^2 의 ^2)
    s2 = _UNIT_EXP_RE.sub("", s2)
    if not s2:
        return None

    if re.fullmatch(r"-?\d+", s2):
        return int(s2)

    # 4) 소수·평문 과학표기
    if re.fullmatch(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", s2):
        try:
            v = float(s2)
        except ValueError:
            return None
        if abs(v) > 1e17:
            return None
        r = round(v)
        return int(r) if abs(v - r) < TOL else NOT_INT
    return None


def _last_number(text: str, require_unique: bool = False) -> int | None:
    """마지막 정수. require_unique면 수 토큰이 정확히 하나일 때만 반환."""
    nums = _NUM_RE.findall(text)
    if require_unique and len(nums) != 1:
        return None
    for tok in reversed(nums):
        v = _to_int(tok)
        if isinstance(v, int):
            return v
    return None


def extract_answer(text: str, tail_chars: int = 400) -> int | None:
    """모델 출력에서 최종 정수 답을 뽑는다. 실패 시 None."""
    if not text:
        return None

    boxed = _find_boxed(text)
    if boxed is not None:
        v = _to_int(boxed)
        if isinstance(v, int):
            return v
        if v is NOT_INT:
            return None                  # 정수가 아님이 확정 -> 폴백 금지
        # 파싱 자체가 실패한 경우에만, 그리고 수가 하나뿐일 때만 폴백
        v = _last_number(boxed, require_unique=True)
        if v is not None:
            return v
        return None                      # boxed가 있는데 못 읽으면 여기서 끝낸다

    tail = text[-tail_chars:]
    m = None
    for m_ in _CLOSER_RE.finditer(tail):
        m = m_
    if m:
        v = _last_number(tail[m.end() : m.end() + 120])
        if v is not None:
            return v

    for line in reversed([l for l in text.strip().splitlines() if l.strip()]):
        v = _last_number(line)
        if v is not None:
            return v
    return None


def _round_boxed(text: str):
    """boxed가 비정수 실수/분수일 때 반올림값. 채점용 추측으로만 쓴다."""
    b = _find_boxed(text)
    if b is None:
        return None
    s = _strip_latex(str(b)).replace(",", "").replace(" ", "")
    fr = _frac_to_value(s)
    if fr is not None:
        try:
            return int(round(float(fr)))
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
    s = _UNIT_EXP_RE.sub("", s)
    if re.fullmatch(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", s):
        try:
            v = float(s)
        except ValueError:
            return None
        return int(round(v)) if abs(v) < 1e17 else None
    return None


def extract_or_fallback(text: str, fallback: int = 0) -> int:
    """제출용. 절대 None을 반환하지 않는다 (빈 값은 오답 처리되므로).

    extract_answer가 None인 경우(비정수 boxed 등)에도 0을 내미는 것보다
    반올림한 값을 내미는 편이 기대값이 높다. 채점은 정수 일치이므로
    어차피 틀릴 확률이 크지만, 0보다 나쁠 이유는 없다.
    """
    v = extract_answer(text)
    if v is not None:
        return v
    v = _round_boxed(text)
    if v is not None:
        return v
    v = _last_number(text[-400:])
    return fallback if v is None else v
