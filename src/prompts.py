"""프롬프트 후보. D-9에서 하나를 고르고 이후 절대 변경하지 않는다.
학습 데이터와 추론이 같은 형식을 공유해야 하기 때문이다.

중요: 베이스 모델이 Qwen2.5-3B-**Instruct** 이므로 chat template을 씌워야 한다.
raw completion으로 넣으면 instruct tuning의 이득을 대부분 버리게 된다.
D-9 게이트 1에서 raw vs chat 을 실측 비교하고, 이후 고정한다.
"""
from typing import List, Dict

SYSTEM = "You are a careful mathematician. You always finish with a final integer answer."

PROMPTS = {
    "A": "{q}",
    "B": "Solve the following math problem step by step. "
         "Put your final answer in \\boxed{{}}.\n\n{q}",
    "C": "Solve the following math problem step by step. "
         "The answer is a single integer. "
         "Put your final answer in \\boxed{{}}.\n\n{q}",
}
DEFAULT = "C"          # D-9 실측 후 확정할 것
USE_CHAT_DEFAULT = True  # D-9 실측 후 확정할 것


def build(q: str, key: str = DEFAULT) -> str:
    """유저 턴 본문만. chat template은 씌우지 않는다."""
    return PROMPTS[key].format(q=q)


def build_messages(q: str, key: str = DEFAULT, system: bool = True) -> List[Dict[str, str]]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": SYSTEM})
    msgs.append({"role": "user", "content": build(q, key)})
    return msgs


def build_prompt(tokenizer, q: str, key: str = DEFAULT,
                 chat: bool = USE_CHAT_DEFAULT, system: bool = True) -> str:
    """vLLM의 llm.generate()에 넣을 최종 문자열.

    chat=True 면 tokenizer의 chat template을 적용한다 (Instruct 모델 정석).
    generate()는 template을 자동으로 씌워주지 않으므로 여기서 명시적으로 처리한다.
    """
    if not chat:
        return build(q, key)
    return tokenizer.apply_chat_template(
        build_messages(q, key, system), tokenize=False, add_generation_prompt=True)
