DEFAULT_PROMPT_ID = "full_persona_v3"

# Kept for backwards compatibility with the existing analysis code.
PROMPT_ID = DEFAULT_PROMPT_ID


VALID_LABELS = {
    "CONTRADICTION": 1,
    "NO_CONTRADICTION": 0,
}


def build_full_persona_v3_prompt(
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


def build_full_persona_robustness_v1_prompt(
    persona: str,
    dialogue: str,
) -> str:
    return f"""Evaluate whether SELF remains consistent with the provided persona.

Choose CONTRADICTION if at least one statement made or clearly implied by SELF in the dialogue is incompatible with a statement in SELF's persona. Otherwise choose NO_CONTRADICTION.

Guidelines:
- A single genuine conflict is enough to choose CONTRADICTION.
- If a persona statement is never addressed in the dialogue, that alone is not a contradiction.
- Only SELF's claims can establish a contradiction with SELF's persona. Statements made by PARTNER may be used as context but not as facts about SELF.
- Take temporal and modal distinctions into account. Past events, current facts, intentions, and wishes conflict only when the information is genuinely incompatible.
- Extra information that can coexist with the persona is not a contradiction.
- Base the decision only on the persona and dialogue provided. Do not infer facts that are not stated or clearly implied.

Persona of SELF:
{persona}

Dialogue:
{dialogue}

Output exactly one of these two labels and no additional text: CONTRADICTION or NO_CONTRADICTION."""


PROMPT_BUILDERS = {
    "full_persona_v3":
        build_full_persona_v3_prompt,
    "full_persona_robustness_v1":
        build_full_persona_robustness_v1_prompt,
}


AVAILABLE_PROMPT_IDS = tuple(
    PROMPT_BUILDERS
)


def build_judge_prompt(
    persona: str,
    dialogue: str,
    prompt_id: str = DEFAULT_PROMPT_ID,
) -> str:
    try:
        builder = PROMPT_BUILDERS[
            prompt_id
        ]
    except KeyError as error:
        raise ValueError(
            f"Unknown prompt ID: {prompt_id!r}. "
            f"Available prompts: {AVAILABLE_PROMPT_IDS}"
        ) from error

    return builder(
        persona=persona,
        dialogue=dialogue,
    )


def parse_judge_output(
    output: str,
) -> tuple[str | None, int | None]:
    """
    Parse a judge label when it appears at the beginning of the output.
    """
    text = str(
        output
    ).strip().upper()

    if text.startswith(
        "NO_CONTRADICTION"
    ):
        return (
            "NO_CONTRADICTION",
            0,
        )

    if text.startswith(
        "CONTRADICTION"
    ):
        return (
            "CONTRADICTION",
            1,
        )

    return None, None