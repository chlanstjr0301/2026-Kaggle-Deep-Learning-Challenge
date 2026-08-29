"""검증기 프롬프트. 학습과 추론이 반드시 같은 문자열을 써야 한다.

점수는 첫 assistant 토큰에서 P(Yes) 대 P(No) 로 읽는다. 이렇게 하면
별도 분류 헤드가 필요 없어 LoRA 만으로 끝나고, vLLM 이 그대로 채점할 수 있다.
(분류 헤드를 붙이면 vLLM 이 못 돌리고 규정상 설명도 번거로워진다.)

stance 는 A/B 대상이다. 심사자 역할을 강하게 주는 편이 나은지 실측으로 정한다.
"""
STANCES = {
    "harsh": "You are a meticulous mathematics grader. You assume every solution "
             "contains an error until proven otherwise, and you check each "
             "computation step before judging.",
    "plain": "You are a mathematics grader.",
}
DEFAULT_STANCE = "harsh"

USER = ("Problem:\n{q}\n\n"
        "Proposed solution:\n{s}\n\n"
        "Is the final answer of this solution correct? "
        "Answer with exactly one word: Yes or No.")

YES, NO = "Yes", "No"


def messages(question: str, solution: str, stance: str = DEFAULT_STANCE):
    return [{"role": "system", "content": STANCES[stance]},
            {"role": "user", "content": USER.format(q=question, s=solution)}]


def build(tok, question: str, solution: str, stance: str = DEFAULT_STANCE) -> str:
    return tok.apply_chat_template(messages(question, solution, stance),
                                   tokenize=False, add_generation_prompt=True)
