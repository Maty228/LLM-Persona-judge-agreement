PROMPT_ID = "full_persona_v3"


VALID_LABELS = {
    "CONTRADICTION": 1,
    "NO_CONTRADICTION": 0,
}


def build_judge_prompt(
    persona: str,
    dialogue: str,
) -> str:
    return f"""You are a binary judge of persona consistency.

Determine whether statements made by SELF in the dialogue contradict at least one statement in SELF's persona.

Rules:
- CONTRADICTION: SELF explicitly states or clearly implies information that cannot be true at the same time as at least one persona statement.
- NO_CONTRADICTION: no such conflict is present.
- One conflicting persona statement is sufficient for CONTRADICTION.
- A persona statement that is not discussed in the dialogue is not a contradiction.
- Claims made only by PARTNER are not evidence about SELF. Use PARTNER's utterances only as conversational context.
- Respect temporal and modal information. Past and present statements, intentions, wishes, and current facts are contradictions only when they are genuinely incompatible.
- Additional compatible information is not a contradiction.
- Use only the information in the persona and dialogue. Do not assume unstated facts.

Persona of SELF:
{persona}

Dialogue:
{dialogue}

Return exactly one label and nothing else: CONTRADICTION or NO_CONTRADICTION."""


def parse_judge_output(
    output: str,
) -> tuple[str | None, int | None]:
    """
    Parse a judge label when it appears at the beginning of the output.
    """
    text = str(output).strip().upper()

    if text.startswith("NO_CONTRADICTION"):
        return "NO_CONTRADICTION", 0

    if text.startswith("CONTRADICTION"):
        return "CONTRADICTION", 1

    return None, None